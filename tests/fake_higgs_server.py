"""Servidor falso com a mesma forma da API documentada do Higgs TTS 3 (POST /v1/audio/speech)."""
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import soundfile as sf

MODE = os.environ.get("FAKE_SEED_MODE", "accept")        # accept | reject | crash


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, b'{"status":"ok"}')
        self._send(404, b"{}")

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            return self._send(404, b"{}")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if MODE == "reject" and "seed" in body:
            return self._send(422, b'{"detail":"Extra inputs are not permitted: seed"}')
        refs = body.get("references") or []
        if not refs or not os.path.exists(refs[0]["audio_path"]):
            return self._send(400, b'{"detail":"reference audio not found"}')
        seed = body.get("seed")
        freq = 220.0 + (seed % 50 if seed is not None else 0)
        sr = 24000
        n = int(sr * max(0.5, len(body["input"]) * 0.06))
        t = np.arange(n) / sr
        wav = (0.2 * np.sin(2 * np.pi * freq * t)).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        self._send(200, buf.getvalue(), "audio/wav",
                   {"X-Max-Tokens": body.get("max_new_tokens"), "X-Seed": seed})


if __name__ == "__main__":
    if MODE == "crash":
        sys.stderr.write("RuntimeError: CUDA out of memory (fake)\n")
        sys.exit(3)
    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
