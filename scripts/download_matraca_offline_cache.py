#!/usr/bin/env python3
"""Baixa o cache recuperável do Matraca Studio para disco local ou Google Drive.

Uso no Colab, após montar o Drive:
  !python download_matraca_offline_cache.py /content/drive/MyDrive/MatracaStudioCache

O script baixa artefatos, mas não instala os pacotes nem altera os modelos ativos.
Execute após o Passo 1 do notebook para que huggingface_hub e whisper estejam disponíveis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


APT_PACKAGES = ("ffmpeg", "sox", "libsox-fmt-all")
PIP_PACKAGES = (
    "sox",
    "onnxruntime",
    "openai-whisper",
    "deep-translator",
    "tensorboardx",
    "webdataset",
    "soxr",
    "soundfile",
    "voxcpm",
    "transformers==4.57.3",
    "azure-cognitiveservices-speech",
    "gradio",
    "requests",
    "numpy",
)
PIP_NO_DEPS_PACKAGES = ("qwen-tts", "omnivoice")
MODEL_REPOS = (
    "openbmb/VoxCPM2",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "k2-fsa/OmniVoice",
)
REPO_FILES = (
    "Matraca_Studio_L4.ipynb",
    "Matraca_Studio_T4.ipynb",
    "higgs_audio_v2_tokenizer.py",
)
REPO_RAW = "https://raw.githubusercontent.com/dgurgel-info/ai-colab-matracastudio/main/{}"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def safe_name(repo: str) -> str:
    return repo.replace("/", "--")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_apt(cache_dir: Path) -> None:
    if shutil.which("apt-get") is None:
        print("⚠️ apt-get não encontrado; ignorando pacotes APT (execute esta etapa no Colab/Linux).")
        return
    apt_dir = cache_dir / "apt-debs"
    apt_dir.mkdir(parents=True, exist_ok=True)
    (apt_dir / "partial").mkdir(exist_ok=True)
    run(["apt-get", "update", "-qq"])
    run([
        "apt-get", "-y", "--download-only",
        "-o", f"Dir::Cache::archives={apt_dir}",
        "install", *APT_PACKAGES,
    ])


def download_pip(cache_dir: Path) -> None:
    pip_dir = cache_dir / "pip-wheels"
    pip_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "pip", "download", "--dest", str(pip_dir),
        "--prefer-binary", *PIP_PACKAGES,
    ])
    # O notebook instala estes dois pacotes sem dependências para manter versões compatíveis.
    run([
        sys.executable, "-m", "pip", "download", "--dest", str(pip_dir),
        "--no-deps", *PIP_NO_DEPS_PACKAGES,
    ])


def download_models(cache_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub não está disponível. Execute primeiro o Passo 1 do notebook."
        ) from error

    model_dir = cache_dir / "huggingface-models"
    for repo_id in MODEL_REPOS:
        target = model_dir / safe_name(repo_id)
        print(f"\n⬇️ Baixando snapshot completo: {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "openai-whisper não está disponível. Execute primeiro o Passo 1 do notebook."
        ) from error
    whisper_dir = model_dir / "whisper-small"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    print("\n⬇️ Baixando Whisper Small")
    whisper.load_model("small", download_root=str(whisper_dir))


def download_repository_files(cache_dir: Path) -> None:
    repo_dir = cache_dir / "notebook-and-helpers"
    repo_dir.mkdir(parents=True, exist_ok=True)
    for filename in REPO_FILES:
        target = repo_dir / filename
        print(f"⬇️ Baixando {filename}")
        urllib.request.urlretrieve(REPO_RAW.format(filename), target)


def write_manifest(cache_dir: Path) -> None:
    files = []
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": str(path.relative_to(cache_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "apt_packages": list(APT_PACKAGES),
        "pip_packages": list(PIP_PACKAGES),
        "pip_no_deps_packages": list(PIP_NO_DEPS_PACKAGES),
        "models": list(MODEL_REPOS) + ["openai/whisper-small"],
        "files": files,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria cache offline do Matraca Studio.")
    parser.add_argument(
        "output_dir",
        help="Destino do cache, por exemplo /content/drive/MyDrive/MatracaStudioCache",
    )
    parser.add_argument("--skip-apt", action="store_true", help="Não baixa os pacotes .deb.")
    parser.add_argument("--skip-models", action="store_true", help="Não baixa os pesos dos modelos.")
    args = parser.parse_args()

    cache_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Cache: {cache_dir}")
    if not args.skip_apt:
        download_apt(cache_dir)
    download_pip(cache_dir)
    if not args.skip_models:
        download_models(cache_dir)
    download_repository_files(cache_dir)
    write_manifest(cache_dir)
    print(f"\n✅ Cache concluído. Manifesto: {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
