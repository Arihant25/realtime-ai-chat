"""Web UI: the voice pipeline runs locally (mic + speakers stay in Python for latency);
the browser shows what it is doing and lets you pick the model for each stage.

    uv run realtime-chat-web          ->  http://127.0.0.1:8890
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from realtime_chat import jan
from realtime_chat.audio import Microphone, Player
from realtime_chat.kokoro_tts import KokoroTTS
from realtime_chat.llm import LLMClient, system_prompt
from realtime_chat.main import MODELS_DIR, Conversation, Listener
from realtime_chat.stt import MODELS as STT_MODELS
from realtime_chat.stt import Parakeet
from realtime_chat.tts import SAMPLE_RATE as TTS_RATE
from realtime_chat.tts import VOICES as ORPHEUS_VOICES
from realtime_chat.tts import OrpheusTTS
from realtime_chat.vad import FRAME, SAMPLE_RATE as MIC_RATE
from realtime_chat.vad import SileroVAD, UtteranceSegmenter, ensure_silero

INDEX = Path(__file__).with_name("web_index.html")
GPU_PROVIDERS = {"CUDAExecutionProvider", "DmlExecutionProvider"}
TTS_KEYS = ("tts", "voice", "speed")
ORPHEUS_ID = "Orpheus-3b-FT-Q4_K_M"
PAGES_ORIGIN = "https://arihant25.github.io"  # the hosted copy of the UI connects to localhost
DEFAULT_LLM = "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M"


def _quiet(msg: str) -> None:
    pass  # the browser shows the conversation; keep the terminal clean


class Engine:
    """Owns the pipeline. Thread-safe: the mic loop, worker threads and the web
    server all go through `lock` when touching components."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_url = args.base_url
        self.allowed_origins = {f"http://{h}:{args.port}" for h in (args.host, "127.0.0.1", "localhost")}
        self.allowed_origins |= {PAGES_ORIGIN, *args.allow_origin}
        self.cfg = {
            "stt": "nemo-parakeet-tdt-0.6b-v2", "stt_gpu": False,
            "llm": args.llm_model,
            # Orpheus (expressive) when it's installed in Jan, else Kokoro (GPU if CUDA works)
            "tts": "orpheus" if jan.installed(ORPHEUS_ID) else
                   "kokoro-gpu" if "CUDAExecutionProvider" in ort.get_available_providers() else "kokoro",
            "voice": "tara" if jan.installed(ORPHEUS_ID) else "af_heart", "speed": 1.0,
            "barge_in": False, "silence_ms": 600, "early_ms": 250,
        }
        self.lock = threading.RLock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.player = Player(TTS_RATE, prebuffer_s=0.1, device=args.output_device)
        self.stt: Parakeet | None = None
        self.tts = None
        self.convo: Conversation | None = None
        self.listener: Listener | None = None
        self.segmenter: UtteranceSegmenter | None = None
        self.vad = SileroVAD(ensure_silero(MODELS_DIR))
        self.status = "loading"
        self.paused = False
        self.mic_level = 0.0
        self.sentences: dict[int, dict] = {}
        self.metrics: dict[str, float] = {}
        self._speech_end: float | None = None  # wall time your last word ended
        self._played_at_final = 0
        self._pending_done: dict | None = None
        self.downloads: dict[str, dict] = {}  # model id -> {"done", "total"}

    # ------------------------------------------------------------------ events
    def broadcast(self, data: dict) -> None:
        if self.loop is None:
            return
        for q in list(self.clients):
            self.loop.call_soon_threadsafe(q.put_nowait, data)

    def on_event(self, kind: str, **data) -> None:
        """Hook for Conversation/Listener. Tracks what the UI needs, then forwards."""
        if kind == "ai_sentence":
            text = data["text"]
            words = text.split()
            # cumulative character position of each word end, to map playback -> words
            ends, pos = [], 0
            for w in words:
                pos += len(w) + 1
                ends.append(pos)
            self.sentences[data["id"]] = {**data, "words": len(words), "ends": ends,
                                          "chars": max(pos, 1), "end": None, "shown": 0}
        elif kind == "ai_sentence_end" and data["id"] in self.sentences:
            self.sentences[data["id"]]["end"] = data["end"]
        elif kind == "user_final":
            self._speech_end = time.monotonic() - self.cfg["silence_ms"] / 1000
            self._played_at_final = self.player.played_total
            self.metrics["stt_ms"] = round(data.get("stt_ms", 0))
        elif kind == "ai_done" and not data.get("interrupted") and self.sentences:
            # generation is done but audio is still playing: tick_loop sends this once it ends
            self._pending_done = data
            return
        elif kind in ("ai_done", "interrupted"):
            self.sentences.clear()
            self._pending_done = None
        self.broadcast({"type": kind, **data})

    # ------------------------------------------------------------------ setup
    def jan_models(self) -> list[str]:
        try:
            r = requests.get(self.base_url.rstrip("/") + "/models", timeout=5)
            return [m["id"] for m in r.json().get("data", [])]
        except requests.RequestException:
            return []

    def options(self) -> dict:
        jan_list = self.jan_models()
        gpu = bool(GPU_PROVIDERS & set(ort.get_available_providers()))
        orpheus = [m for m in jan_list if "orpheus" in m.lower()]
        tts = [{"id": "kokoro", "label": "Kokoro-82M · CPU"}]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            tts.append({"id": "kokoro-gpu", "label": "Kokoro-82M · GPU"})
        tts.append({"id": "orpheus", "label": "Orpheus 3B · GPU via Jan, expressive",
                    "install": None if orpheus else "Orpheus-3b-FT-Q4_K_M"})
        llms = [{"id": m, "label": m} for m in jan_list
                if "orpheus" not in m.lower() and "sentence-transformer" not in m]
        # catalog models that aren't in Jan yet: picking one downloads it into Jan
        on_disk = {e["id"] for e in jan.CATALOG if jan.installed(e["id"])}
        for e in jan.CATALOG:
            if e["kind"] == "llm" and e["id"] not in jan_list:
                llms.append({"id": e["id"], "label": e["label"],
                             "install": e["id"] if e["id"] not in on_disk else None,
                             "restart": e["id"] in on_disk})
        kokoro_voices = self.tts.kokoro.get_voices() if isinstance(self.tts, KokoroTTS) else []
        return {
            "stt": [{"id": k, "label": v} for k, v in STT_MODELS.items()],
            "stt_gpu": gpu,
            "llm": llms,
            "tts": tts,
            "voices": {"kokoro": kokoro_voices or ["af_heart"], "kokoro-gpu": kokoro_voices or ["af_heart"],
                       "orpheus": ORPHEUS_VOICES},
            "jan": {"online": bool(jan_list), "restartable": jan.is_installed(),
                    "models_max": jan.get_models_max()},
            "downloads": {k: v for k, v in self.downloads.items()},
        }

    def install(self, model_id: str) -> None:
        entry = next((e for e in jan.CATALOG if e["id"] == model_id), None)
        if entry is None or model_id in self.downloads:
            return
        self.downloads[model_id] = {"done": 0, "total": 0}
        last = [0.0]

        def progress(done: int, total: int) -> None:
            self.downloads[model_id] = {"done": done, "total": total}
            if time.monotonic() - last[0] > 0.5:
                last[0] = time.monotonic()
                self.broadcast({"type": "download", "id": model_id, "done": done, "total": total})

        try:
            jan.install(entry, progress)
            self.broadcast({"type": "download_done", "id": model_id, "label": entry["label"]})
        except Exception as e:
            self.broadcast({"type": "error", "text": f"Download of {entry['label']} failed: {e}"})
        finally:
            self.downloads.pop(model_id, None)
            self.broadcast({"type": "config", "config": self.cfg, "options": self.options()})

    def restart_jan(self, models_max: int | None) -> None:
        prev = self.status
        self.status = "loading"
        if self.convo:
            self.convo.interrupt()
        self.broadcast({"type": "status", "text": "Restarting Jan…"})
        try:
            jan.restart(self.base_url, models_max)
            self.broadcast({"type": "status", "text": ""})
        except Exception as e:
            self.broadcast({"type": "error", "text": str(e)})
        self.status = "ready" if prev == "ready" or self.listener else prev
        self.broadcast({"type": "config", "config": self.cfg, "options": self.options()})

    def _make_tts(self):
        c = self.cfg
        if c["tts"] == "orpheus":
            voice = c["voice"] if c["voice"] in ORPHEUS_VOICES else "tara"
            c["voice"] = voice
            model = next((m for m in self.jan_models() if "orpheus" in m.lower()), self.args.tts_model)
            return OrpheusTTS(self.base_url, model, voice=voice)
        tts = KokoroTTS(MODELS_DIR, voice="af_heart", speed=c["speed"], gpu=c["tts"] == "kokoro-gpu")
        if c["voice"] in tts.kokoro.get_voices():
            tts.voice = c["voice"]
        else:
            c["voice"] = tts.voice
        return tts

    def apply(self, changes: dict | None = None) -> None:
        """(Re)build whatever the changed settings affect. Runs on a worker thread."""
        changes = changes or {}
        first = self.convo is None
        with self.lock:
            old = dict(self.cfg)
            self.cfg.update({k: v for k, v in changes.items() if k in self.cfg})
            changed = {k for k in self.cfg if self.cfg[k] != old[k]} | ({"*"} if first else set())
            if not changed:
                return
            self.status = "loading"
            self.broadcast({"type": "status", "text": "Loading models…"})
            if self.convo:
                self.convo.interrupt()
            try:
                if first or {"stt", "stt_gpu"} & changed:
                    self.broadcast({"type": "status", "text": f"Loading {STT_MODELS.get(self.cfg['stt'], self.cfg['stt'])}…"})
                    self.stt = Parakeet(gpu=self.cfg["stt_gpu"], model=self.cfg["stt"])
                if first or "tts" in changed or ("voice" in changed and self.cfg["tts"] == "orpheus"):
                    self.broadcast({"type": "status", "text": "Loading TTS…"})
                    self.tts = self._make_tts()
                elif isinstance(self.tts, KokoroTTS):
                    self.tts.voice = self.cfg["voice"]
                    self.tts.speed = self.cfg["speed"]
                llm = LLMClient(self.base_url, self.cfg["llm"])
                prompt = system_prompt(sound_tags=self.cfg["tts"] == "orpheus")
                if self.convo is None:
                    self.convo = Conversation(llm, self.tts, self.player, prompt, on_event=self.on_event,
                                              log=_quiet)
                else:
                    self.convo.llm, self.convo.tts = llm, self.tts
                    self.convo.history[0] = {"role": "system", "content": prompt}
                self.segmenter = UtteranceSegmenter(self.vad, min_silence_ms=self.cfg["silence_ms"],
                                                    early_silence_ms=self.cfg["early_ms"])
                self.listener = Listener(self.segmenter, self.stt, self.convo, self.player,
                                         barge_in=self.cfg["barge_in"], on_event=self.on_event,
                                         log=_quiet)
                self.status = "ready"
                self.broadcast({"type": "status", "text": ""})
            except Exception as e:  # keep the old config working if a new model fails
                self.cfg = old
                self.status = "ready" if self.listener else "error"
                self.broadcast({"type": "error", "text": f"Couldn't load: {e}"})
            self.broadcast({"type": "config", "config": self.cfg, "options": self.options()})

    # ------------------------------------------------------------------ loops
    def mic_loop(self) -> None:
        with Microphone(MIC_RATE, FRAME, device=self.args.input_device) as mic:
            while True:
                frame = mic.frames.get()
                self.mic_level = float(np.sqrt(np.mean(frame**2)))
                if self.paused or self.status != "ready":
                    continue
                with self.lock:
                    if self.listener.handle(frame) == "unmuted":
                        while not mic.frames.empty():
                            mic.frames.get_nowait()

    def partial_loop(self) -> None:
        """Re-transcribe the utterance so far while you talk, for live captions."""
        while True:
            time.sleep(0.35)
            seg, stt = self.segmenter, self.stt
            # only while you are actively talking: once you pause, the CPU goes to the
            # real transcription and the reply
            if (self.status != "ready" or seg is None or not seg.in_speech
                    or len(seg.frames) < 10 or seg.silent_run > 3):
                continue
            audio = np.concatenate(list(seg.frames))
            try:
                text = stt.transcribe(audio, MIC_RATE)
            except Exception:
                continue
            if text and seg.in_speech:
                self.broadcast({"type": "user_partial", "text": text})

    def state(self) -> str:
        if self.status != "ready":
            return self.status
        if self.player.level > 0.002 or self.player.busy():
            return "speaking"
        if self.convo and self.convo.active():
            return "thinking"
        if self.paused:
            return "paused"
        if self.segmenter and self.segmenter.in_speech:
            return "user"
        return "listening"

    def tick_loop(self) -> None:
        """~20 Hz: state + levels, words as they are played, latency."""
        while True:
            time.sleep(0.05)
            played = self.player.played_total
            for s in list(self.sentences.values()):
                total = (s["end"] - s["start"]) if s["end"] else max(self.player.fed_total - s["start"], s["expected"])
                frac = min(max((played - s["start"]) / max(total, 1), 0.0), 1.0)
                n = sum(1 for e in s["ends"] if e <= frac * s["chars"] + 2)
                if n > s["shown"]:
                    s["shown"] = n
                    self.broadcast({"type": "ai_words", "id": s["id"], "n": n})
            if self._pending_done and not self.player.busy():  # last audio has played
                for s in self.sentences.values():
                    self.broadcast({"type": "ai_words", "id": s["id"], "n": s["words"]})
                self.sentences.clear()
                self.broadcast({"type": "ai_done", **self._pending_done})
                self._pending_done = None
            if self._speech_end is not None and played > self._played_at_final:
                self.metrics["voice_to_voice_ms"] = round((time.monotonic() - self._speech_end) * 1000)
                self._speech_end = None
                self.broadcast({"type": "metrics", **self.metrics})
            self.broadcast({"type": "tick", "state": self.state(),
                            "mic": round(self.mic_level, 4), "out": round(self.player.level, 4)})

    def command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "config":
            threading.Thread(target=self.apply, args=(msg.get("config", {}),), daemon=True).start()
        elif kind == "pause":
            self.paused = bool(msg.get("paused"))
            if self.paused and self.convo:
                self.convo.interrupt()
            self.broadcast({"type": "paused", "paused": self.paused})  # keep every open page in sync
        elif kind == "stop" and self.convo:
            self.convo.interrupt()
            self.broadcast({"type": "interrupted"})
        elif kind == "install":
            threading.Thread(target=self.install, args=(msg.get("id"),), daemon=True).start()
        elif kind == "restart_jan":
            mm = msg.get("models_max")
            threading.Thread(target=self.restart_jan, args=(None if mm is None else int(mm),),
                             daemon=True).start()
        elif kind == "reset" and self.convo:
            with self.lock:
                self.convo.interrupt()
                del self.convo.history[1:]
            self.broadcast({"type": "reset"})


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="realtime-ai-chat")

    @app.on_event("startup")
    async def startup():
        engine.loop = asyncio.get_running_loop()
        threading.Thread(target=engine.apply, daemon=True).start()
        for target in (engine.mic_loop, engine.partial_loop, engine.tick_loop):
            threading.Thread(target=target, daemon=True).start()

    @app.get("/")
    def index():
        return HTMLResponse(INDEX.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        # Browsers let any website open a WebSocket to localhost, and this socket can read
        # your conversation and control the mic. Only accept our own pages.
        origin = sock.headers.get("origin")
        if origin is not None and origin not in engine.allowed_origins:
            await sock.close(code=1008)
            return
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        engine.clients.add(q)
        history = [{"role": m["role"], "text": m["content"]}
                   for m in (engine.convo.history[1:] if engine.convo else [])]
        await sock.send_json({"type": "hello", "config": engine.cfg, "history": history,
                              "options": engine.options() if engine.convo else None,
                              "metrics": engine.metrics, "paused": engine.paused})

        async def sender():
            while True:
                await sock.send_json(await q.get())

        task = asyncio.create_task(sender())
        try:
            while True:
                engine.command(await sock.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            engine.clients.discard(q)

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Web UI for realtime-ai-chat")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8890)
    p.add_argument("--base-url", default="http://127.0.0.1:1337/v1")
    p.add_argument("--llm-model", default=DEFAULT_LLM)
    p.add_argument("--tts-model", default="Orpheus-3b-FT-Q4_K_M")
    p.add_argument("--input-device", default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--allow-origin", action="append", default=[],
                   help="extra web origin allowed to connect, e.g. a fork's GitHub Pages URL")
    args = p.parse_args()
    for attr in ("input_device", "output_device"):
        v = getattr(args, attr)
        if v is not None and v.isdigit():
            setattr(args, attr, int(v))

    url = f"http://{args.host}:{args.port}"
    print(f"Web UI on {url}")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(Engine(args)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
