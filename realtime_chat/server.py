"""OpenAI-compatible speech server, so apps like Open WebUI can use Parakeet and Orpheus.

    POST /v1/audio/transcriptions   multipart `file` (any format ffmpeg reads) -> {"text": ...}
    POST /v1/audio/speech           {"input": ..., "voice": "tara"} -> audio/wav

Run:  uv run realtime-chat-server   (listens on http://127.0.0.1:8880/v1)
"""

from __future__ import annotations

import argparse
import io
import threading
import wave

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from realtime_chat.kokoro_tts import KokoroTTS
from realtime_chat.llm import clean_for_speech
from realtime_chat.main import MODELS_DIR
from realtime_chat.stt import Parakeet
from realtime_chat.tts import SAMPLE_RATE as TTS_RATE
from realtime_chat.tts import VOICES, OrpheusTTS

STT_RATE = 16000


def decode_audio(data: bytes) -> np.ndarray:
    """Any container/codec -> 16 kHz mono float32."""
    with av.open(io.BytesIO(data)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=STT_RATE)
        chunks = [r.to_ndarray().reshape(-1)
                  for frame in container.decode(audio=0) for r in resampler.resample(frame)]
        chunks += [r.to_ndarray().reshape(-1) for r in resampler.resample(None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def to_wav(audio: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class SpeechRequest(BaseModel):
    input: str
    voice: str = "tara"
    model: str | None = None
    response_format: str | None = None  # always WAV; browsers play it regardless
    speed: float | None = None


def create_app(stt: Parakeet, tts: OrpheusTTS | KokoroTTS) -> FastAPI:
    app = FastAPI(title="realtime-ai-chat speech server")
    stt_lock = threading.Lock()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "parakeet-tdt-0.6b-v2", "object": "model"},
                                           {"id": "orpheus", "object": "model"}]}

    @app.get("/v1/audio/voices")
    def voices():
        return {"voices": tts.kokoro.get_voices() if isinstance(tts, KokoroTTS) else VOICES}

    @app.post("/v1/audio/transcriptions")
    def transcribe(file: UploadFile = File(...), model: str = Form(None),
                   response_format: str = Form("json")):
        try:
            audio = decode_audio(file.file.read())
        except av.FFmpegError as e:
            raise HTTPException(400, f"Can't decode audio: {e}")
        with stt_lock:
            text = stt.transcribe(audio, STT_RATE) if len(audio) > STT_RATE // 10 else ""
        if response_format == "text":
            return Response(text, media_type="text/plain")
        return {"text": text}

    @app.post("/v1/audio/speech")
    def speech(req: SpeechRequest):
        text = clean_for_speech(req.input)
        if not text:
            raise HTTPException(400, "Empty input")
        if isinstance(tts, KokoroTTS):
            voices = tts.kokoro.get_voices()
            voice = req.voice if req.voice in voices else tts.voice
            audio, _ = tts.kokoro.create(text, voice=voice, speed=req.speed or tts.speed)
            chunks = [audio.astype(np.float32)]
        else:
            voice = req.voice if req.voice in VOICES else tts.voice
            engine = OrpheusTTS(tts.url.rsplit("/", 1)[0], tts.model, voice=voice,
                                decoder=tts.decoder,
                                api_key=tts.headers.get("Authorization", "")[7:])
            chunks = list(engine.synthesize(text, threading.Event()))
        audio = np.concatenate(chunks) if chunks else np.zeros(TTS_RATE // 10, np.float32)
        return Response(to_wav(audio, TTS_RATE), media_type="audio/wav")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI-compatible Parakeet STT + Orpheus TTS server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8880)
    p.add_argument("--base-url", default="http://127.0.0.1:1337/v1", help="Jan API (serves Orpheus)")
    p.add_argument("--api-key", default="")
    p.add_argument("--tts", default="kokoro", choices=["kokoro", "orpheus"])
    p.add_argument("--tts-model", default="Orpheus-3b-FT-Q4_K_M")
    p.add_argument("--voice", default=None, help="default voice (af_heart / tara)")
    p.add_argument("--stt-gpu", action="store_true")
    args = p.parse_args()

    stt = Parakeet(gpu=args.stt_gpu)
    if args.tts == "kokoro":
        tts = KokoroTTS(MODELS_DIR, voice=args.voice or "af_heart")
    else:
        tts = OrpheusTTS(args.base_url, args.tts_model, voice=args.voice or "tara",
                         api_key=args.api_key)
    print(f"Speech API on http://{args.host}:{args.port}/v1")
    uvicorn.run(create_app(stt, tts), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
