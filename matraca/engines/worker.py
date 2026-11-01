"""Worker de um motor de voz: roda DENTRO do venv do motor e expõe o modelo por HTTP local.

    python -m matraca.engines.worker --engine voxcpm2 --port 8123

Cada motor tem o seu venv (bibliotecas e versões próprias), por isso nenhum import pesado fica no topo:
os carregadores abaixo só importam torch/voxcpm/qwen_tts/omnivoice quando o worker já está no venv certo.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict

from matraca.engines import rpc

LOADERS: Dict[str, Callable[[], Any]] = {}


def loader(name: str):
    def register(fn):
        LOADERS[name] = fn
        return fn
    return register


# ------------------------------------------------------------------ carregadores (um por motor)
@loader("voxcpm2")
def load_voxcpm2():
    import torch
    from voxcpm import VoxCPM
    print("🎙️ Carregando VoxCPM2 2B (OpenBMB) — 48 kHz / qualidade máxima...", flush=True)
    return VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        optimize=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@loader("qwen3")
def load_qwen3():
    import torch
    from qwen_tts import Qwen3TTSModel
    cuda = torch.cuda.is_available()
    # BF16 emulado não significa suporte nativo (Tesla T4: compute capability 7.5).
    dtype = torch.bfloat16 if cuda and torch.cuda.get_device_capability()[0] >= 8 else (
        torch.float16 if cuda else torch.float32)
    print(f"🤖 Carregando Qwen3-TTS 1.7B Base (Alibaba Cloud), precisão {dtype}...", flush=True)
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                          device_map="cuda:0" if cuda else "cpu", dtype=dtype)
    cfg = getattr(model.model, "generation_config", None)
    if cfg is not None:
        cfg.pad_token_id = cfg.eos_token_id
    defaults = getattr(model, "generate_defaults", None)
    if defaults is not None and hasattr(defaults, "pad_token_id") and cfg is not None:
        defaults.pad_token_id = cfg.eos_token_id
    return model


@loader("omnivoice")
def load_omnivoice():
    import torch
    from omnivoice import OmniVoice
    cuda = torch.cuda.is_available()
    print("🎙️ Carregando OmniVoice (k2-fsa)...", flush=True)
    return OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0" if cuda else "cpu",
                                     dtype=torch.float16 if cuda else torch.float32)


# Motores cujas chamadas rodam em torch.inference_mode (como o notebook sempre fez com o Qwen3-TTS).
INFERENCE_MODE = {"qwen3"}


@contextlib.contextmanager
def seeded(seed):
    """Mesma regra do notebook: fixa a semente só durante a chamada e restaura o RNG depois."""
    if seed is None:
        yield
        return
    import torch
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        for gpu_id in devices:
            torch.cuda.default_generators[gpu_id].manual_seed(int(seed))
        yield


class Service:
    """Estado do worker: o modelo, as referências a objetos remotos e a trava de uso (um pedido por vez)."""

    def __init__(self, engine: str, model: Any):
        self.engine, self.model = engine, model
        self.refs: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _resolve(self, path: str):
        target = self.model
        for part in filter(None, path.split(".")):
            target = getattr(target, part)
        return target

    def handle(self, request: dict) -> dict:
        op = request.get("op", "call")
        with self.lock:
            target = self._resolve(request.get("method", ""))
            if op == "attr":
                return {"result": rpc.encode(target, self.refs)}
            args = rpc.decode(request.get("args", []), self.refs)
            kwargs = rpc.decode(request.get("kwargs", {}), self.refs)
            with contextlib.ExitStack() as stack:
                if self.engine in INFERENCE_MODE:
                    import torch
                    stack.enter_context(torch.inference_mode())
                stack.enter_context(seeded(request.get("seed")))
                result = target(*args, **kwargs)
            return {"result": rpc.encode(result, self.refs)}


def make_handler(service: Service, shutdown: Callable[[], None]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):          # silencioso: o log do worker é o stdout do modelo
            pass

        def _send(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ready": True, "engine": service.engine})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                shutdown()
                return
            if self.path != "/rpc":
                self._send(404, {"error": "not found"})
                return
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, service.handle(request))
            except BaseException as exc:       # o erro volta ao notebook com o traceback do worker
                self._send(500, {"error": type(exc).__name__, "message": str(exc),
                                 "traceback": traceback.format_exc()[-3000:]})

    return Handler


def serve(engine: str, port: int, model: Any = None) -> None:
    model = LOADERS[engine]() if model is None else model
    service = Service(engine, model)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service, lambda: threading.Thread(
        target=server.shutdown, daemon=True).start()))
    print(f"✅ Worker {engine} pronto na porta {port}.", flush=True)
    server.serve_forever()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=sorted(LOADERS))
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    serve(args.engine, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
