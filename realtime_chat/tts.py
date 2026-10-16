"""Streaming Orpheus TTS: raw completions from a llama.cpp-style server, decoded with SNAC.

Orpheus emits audio as <custom_token_N> text tokens, 7 per SNAC frame. Every new
frame we decode a sliding window of the last 4 frames and keep one frame's worth
of audio (2048 samples at 24 kHz, ~85 ms), which avoids clicks at window edges.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from collections.abc import Iterator

import numpy as np
import onnxruntime as ort
import requests
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 24000
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
FRAME_SAMPLES = 2048
WINDOW = 28  # tokens = 4 SNAC frames


class SnacDecoder:
    """SNAC 24 kHz decoder as ONNX (onnx-community/snac_24khz-ONNX), so no PyTorch needed.

    Runs on the GPU via DirectML when available (~5x faster than CPU); otherwise on one
    CPU thread, because llama.cpp samples Orpheus tokens on the CPU and starving it
    costs more throughput than a slower decode.
    """

    def __init__(self, repo: str = "onnx-community/snac_24khz-ONNX",
                 filename: str = "onnx/decoder_model.onnx", gpu: bool = True):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if gpu and "CUDAExecutionProvider" in available:
            providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        elif gpu and "DmlExecutionProvider" in available:
            providers.insert(0, "DmlExecutionProvider")
            opts.enable_mem_pattern = False  # required by DirectML
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3  # hide "some nodes were assigned to CPU" warnings
        self.session = ort.InferenceSession(hf_hub_download(repo, filename), sess_options=opts,
                                            providers=providers)
        self.device = self.session.get_providers()[0]
        self._lock = threading.Lock()  # DirectML sessions aren't safe for concurrent runs

    def decode(self, codes: list[int]) -> np.ndarray | None:
        """Decode a multiple of 7 code ids into float32 audio."""
        n = len(codes) // 7
        f = np.asarray(codes[: n * 7], dtype=np.int64).reshape(n, 7)
        c0 = f[:, 0]
        c1 = f[:, [1, 4]].reshape(-1)
        c2 = f[:, [2, 3, 5, 6]].reshape(-1)
        if any(((c < 0) | (c > 4096)).any() for c in (c0, c1, c2)):
            return None
        with self._lock:
            (audio,) = self.session.run(None, {"audio_codes.0": c0[None], "audio_codes.1": c1[None],
                                               "audio_codes.2": c2[None]})
        return audio[0, 0].astype(np.float32)


class OrpheusTTS:
    def __init__(self, base_url: str, model: str, voice: str = "tara", api_key: str = "",
                 temperature: float = 0.6, top_p: float = 0.9, decoder: SnacDecoder | None = None):
        if voice not in VOICES:
            raise ValueError(f"Unknown voice {voice!r}; choose from {', '.join(VOICES)}")
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.voice = voice
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.temperature = temperature
        self.top_p = top_p
        self.decoder = decoder or SnacDecoder()

    def _tokens(self, text: str, cancel: threading.Event) -> Iterator[int]:
        payload = {
            "model": self.model,
            "prompt": f"<|audio|>{self.voice}: {text}<|eot_id|>",
            "max_tokens": 4096,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": 1.1,  # Orpheus is unstable without it
            "stream": True,
        }
        with requests.post(self.url, headers=self.headers, json=payload, stream=True,
                           timeout=120) as r:
            r.raise_for_status()
            pending = ""
            index = 0
            for line in r.iter_lines(decode_unicode=True):
                if cancel.is_set():
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                pending += json.loads(data)["choices"][0].get("text") or ""
                last = 0
                for m in _TOKEN_RE.finditer(pending):
                    code = int(m.group(1)) - 10 - (index % 7) * 4096
                    last = m.end()
                    if code > 0:
                        index += 1
                        yield code
                pending = pending[last:]

    def _tokens_threaded(self, text: str, cancel: threading.Event) -> Iterator[int]:
        """Read the token stream on its own thread so decoding never stalls the socket."""
        q: queue.Queue = queue.Queue()

        def reader():
            try:
                for code in self._tokens(text, cancel):
                    q.put(code)
            except Exception as e:  # surfaced in the consumer
                q.put(e)
            finally:
                q.put(None)

        threading.Thread(target=reader, daemon=True).start()
        while (item := q.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def synthesize(self, text: str, cancel: threading.Event, hop: int = 8) -> Iterator[np.ndarray]:
        """Yield float32 audio chunks at 24 kHz as soon as they are decodable.

        Frames are emitted with 1 frame of left and 2 of right context (the
        reference Orpheus window). The first chunk is a single frame for low
        latency; chunk sizes then double up to `hop` frames per decode, which
        costs far less CPU than decoding a 4-frame window for every frame.
        """
        codes: list[int] = []
        nxt = 1  # next frame to emit; frame 0 has no left context
        step = 1
        for code in self._tokens_threaded(text, cancel):
            codes.append(code)
            if len(codes) % 7:
                continue
            if len(codes) // 7 >= nxt + step + 2:
                audio = self._frames(codes, nxt - 1, nxt + step + 2)
                if audio is not None:
                    yield audio[FRAME_SAMPLES : FRAME_SAMPLES * (step + 1)]
                nxt += step
                step = min(step * 2, hop)  # ramp 1, 2, 4, ... so early audio arrives steadily
        if cancel.is_set():
            return
        # flush whatever the windowing hasn't emitted yet
        n = len(codes) // 7
        if n > nxt:
            audio = self._frames(codes, nxt - 1, n)
            if audio is not None:
                yield audio[FRAME_SAMPLES:]

    def _frames(self, codes: list[int], start: int, end: int) -> np.ndarray | None:
        return self.decoder.decode(codes[start * 7 : end * 7])
