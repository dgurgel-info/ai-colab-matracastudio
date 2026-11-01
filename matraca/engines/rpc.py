"""Protocolo entre o notebook e os workers dos motores (um processo por venv).

Mensagens são JSON; arrays numpy/tensores viajam em base64 (``__t__``). Objetos que não cabem em JSON
(ex.: o prompt de voz do Qwen3-TTS) ficam no worker e voltam como referência opaca (``RemoteRef``).
Só biblioteca padrão + numpy, para funcionar tanto no notebook quanto dentro de cada venv.
"""

from __future__ import annotations

import base64
import contextvars
from typing import Any, Dict, Optional

import numpy as np

TAG = "__t__"

# Semente da geração em curso. O ``seeded_generation`` do notebook a define; o proxy a repassa ao worker,
# que a aplica no processo onde o modelo realmente roda (o RNG do notebook não afeta o worker).
ACTIVE_SEED: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("matraca_seed", default=None)


class RemoteRef:
    """Referência a um objeto que vive no worker."""

    def __init__(self, ref_id: str, type_name: str = "object"):
        self.id, self.type_name = ref_id, type_name

    def __repr__(self) -> str:
        return f"<RemoteRef {self.type_name}#{self.id}>"


def _is_tensor(obj: Any) -> bool:
    return type(obj).__module__.startswith("torch") and hasattr(obj, "detach")


def _pack(arr: np.ndarray, kind: str) -> Dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    return {TAG: kind, "dtype": str(arr.dtype), "shape": list(arr.shape),
            "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def encode(obj: Any, refs: Optional[Dict[str, Any]] = None) -> Any:
    """Python -> JSON. ``refs`` (lado do worker) guarda os objetos não serializáveis."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _pack(obj, "np")
    if _is_tensor(obj):
        return _pack(obj.detach().cpu().float().numpy(), "torch")
    if isinstance(obj, RemoteRef):
        return {TAG: "ref", "id": obj.id, "type": obj.type_name}
    if isinstance(obj, tuple):
        return {TAG: "tuple", "items": [encode(i, refs) for i in obj]}
    if isinstance(obj, list):
        return [encode(i, refs) for i in obj]
    if isinstance(obj, dict):
        return {str(k): encode(v, refs) for k, v in obj.items()}
    if refs is not None:                       # worker: guarda o objeto e devolve só a referência
        ref_id = str(len(refs) + 1)
        refs[ref_id] = obj
        return {TAG: "ref", "id": ref_id, "type": type(obj).__name__}
    raise TypeError(f"Tipo não serializável no protocolo do worker: {type(obj).__name__}")


def decode(obj: Any, refs: Optional[Dict[str, Any]] = None) -> Any:
    """JSON -> Python. No worker, ``refs`` resolve referências de volta aos objetos reais."""
    if isinstance(obj, list):
        return [decode(i, refs) for i in obj]
    if isinstance(obj, dict):
        kind = obj.get(TAG)
        if kind in ("np", "torch"):
            arr = np.frombuffer(base64.b64decode(obj["data"]), dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
            arr = arr.copy()                   # frombuffer é somente leitura
            if kind == "torch":
                try:
                    import torch
                    return torch.from_numpy(arr)
                except ImportError:
                    return arr
            return arr
        if kind == "tuple":
            return tuple(decode(i, refs) for i in obj["items"])
        if kind == "ref":
            if refs is not None:
                if obj["id"] not in refs:
                    raise KeyError(f"Referência remota inexistente: {obj['id']} (o worker foi reiniciado?)")
                return refs[obj["id"]]
            return RemoteRef(obj["id"], obj.get("type", "object"))
        return {k: decode(v, refs) for k, v in obj.items()}
    return obj
