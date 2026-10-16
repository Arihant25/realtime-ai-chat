"""Real-time local speech-to-speech chat.

mic -> Silero VAD -> Parakeet (STT) -> local LLM via Jan -> Kokoro or Orpheus (TTS) -> speakers
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import onnxruntime as ort
import requests
import sounddevice as sd

from realtime_chat.audio import Microphone, Player
from realtime_chat.kokoro_tts import VOICES as KOKORO_VOICES
from realtime_chat.kokoro_tts import KokoroTTS
from realtime_chat.llm import LLMClient, sentences, system_prompt
from realtime_chat.stt import Parakeet
from realtime_chat.tts import SAMPLE_RATE as TTS_RATE  # Kokoro is 24 kHz too
from realtime_chat.tts import VOICES as ORPHEUS_VOICES
from realtime_chat.tts import OrpheusTTS
from realtime_chat.vad import FRAME, SAMPLE_RATE as MIC_RATE
from realtime_chat.vad import SileroVAD, UtteranceSegmenter, ensure_silero

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ECHO_TAIL_S = 0.4  # ignore the mic this long after the assistant stops (half-duplex mode)


def _no_event(kind: str, **data) -> None:
    pass


def _print(msg: str) -> None:
    print(msg, flush=True)


class Conversation:
    def __init__(self, llm: LLMClient, tts: OrpheusTTS, player: Player, system_prompt: str,
                 tts_parallel: int = 2, on_event=_no_event, log=_print):
        self.llm = llm
        self.tts = tts
        self.player = player
        self.tts_parallel = tts_parallel
        self.emit = on_event  # (kind, **data) -> None, for UIs; called from worker threads
        self.log = log  # transcript lines for the terminal app; the web UI silences it
        self._sentence_id = 0
        self.history = [{"role": "system", "content": system_prompt}]
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._speculative = False
        self._turn_start = len(self.history)
        # running estimates used to size each sentence's playback buffer
        self._rtf = 0.8  # Orpheus audio seconds per wall second
        self._chars_per_s = 14.0  # speaking rate

    def confirm(self) -> None:
        self._speculative = False
        self.player.set_gate(False)

    def retract(self) -> None:
        self.interrupt()
        self._speculative = False
        self.player.set_gate(False)
        del self.history[self._turn_start :]

    def active(self) -> bool:
        """True while a confirmed reply is being generated or played."""
        running = self._thread is not None and self._thread.is_alive() and not self._speculative
        return running or self.player.busy()

    def interrupt(self) -> None:
        self._cancel.set()
        self.player.clear()
        if self._thread is not None:
            self._thread.join()
        self.player.clear()

    def reply_to(self, text: str, speculative: bool = False) -> None:
        """Start replying. A speculative reply generates audio but holds playback
        until confirm(); retract() drops it from the conversation entirely."""
        self.interrupt()
        self._cancel = threading.Event()
        self._speculative = speculative
        self.player.set_gate(speculative)
        self._turn_start = len(self.history)
        self.history.append({"role": "user", "content": text})
        self._thread = threading.Thread(target=self._turn, args=(self._cancel,), daemon=True)
        self._thread.start()

    def _turn(self, cancel: threading.Event) -> None:
        """LLM sentences are synthesized up to `tts_parallel` at a time (the server
        batches them, ~1.75x throughput for 2) and played strictly in order."""
        spoken: list[str] = []
        slots = threading.Semaphore(self.tts_parallel)
        order: queue.Queue[tuple[str, queue.Queue] | None] = queue.Queue()

        def synth(sentence: str, out: queue.Queue) -> None:
            t0, samples = time.perf_counter(), 0
            try:
                for chunk in self.tts.synthesize(sentence, cancel):
                    samples += len(chunk)
                    out.put(chunk)
                if samples and not cancel.is_set():
                    dur = samples / TTS_RATE
                    self._rtf = 0.7 * self._rtf + 0.3 * dur / (time.perf_counter() - t0)
                    self._chars_per_s = 0.7 * self._chars_per_s + 0.3 * len(sentence) / dur
            except Exception as e:  # network errors, or a TTS model failing on some input
                self.log(f"  [TTS error: {e}]")
                self.emit("error", text=f"TTS error: {e}")
            finally:
                out.put(None)
                slots.release()

        def play() -> None:
            for sentence, out in iter(order.get, None):
                while self._speculative and not cancel.is_set():
                    time.sleep(0.01)  # audio keeps accumulating in `out` meanwhile
                if not cancel.is_set():
                    self.log(f"  AI: {sentence}")
                # Orpheus can run below real time, so hold back just enough of this
                # sentence that the rest arrives before playback catches up.
                expected_s = len(sentence) / self._chars_per_s
                need = int(max(0.0, expected_s * (1 - self._rtf) * 1.15) * TTS_RATE)
                held, got, done = [], 0, False
                while got < need:
                    chunk = out.get()
                    if chunk is None:
                        done = True
                        break
                    held.append(chunk)
                    got += len(chunk)
                self._sentence_id += 1
                sid = self._sentence_id
                if not cancel.is_set():
                    self.emit("ai_sentence", id=sid, text=sentence, start=self.player.fed_total,
                              expected=int(expected_s * TTS_RATE))
                for c in held:
                    if not cancel.is_set():
                        self.player.feed(c)
                while not done:  # arrays can't be compared to None, hence no iter(get, None)
                    chunk = out.get()
                    if chunk is None:
                        break
                    if not cancel.is_set():
                        self.player.feed(chunk)
                if not cancel.is_set():
                    spoken.append(sentence)
                    self.emit("ai_sentence_end", id=sid, end=self.player.fed_total)

        player = threading.Thread(target=play, daemon=True)
        player.start()
        try:
            for sentence in sentences(self.llm.stream(list(self.history), cancel)):
                slots.acquire()
                if cancel.is_set():
                    slots.release()
                    break
                out: queue.Queue = queue.Queue()
                order.put((sentence, out))
                threading.Thread(target=synth, args=(sentence, out), daemon=True).start()
        except requests.RequestException as e:
            self.log(f"  [LLM error: {e}]")
            self.emit("error", text=f"LLM error: {e}")
        finally:
            order.put(None)
            player.join()
            self.player.finish()

        reply = " ".join(spoken)
        if cancel.is_set():
            reply += " [interrupted by user]" if reply else "[interrupted by user]"
        self.history.append({"role": "assistant", "content": reply})
        self.emit("ai_done", interrupted=cancel.is_set())


class Listener:
    """Turn-taking: mic frames -> VAD events -> transcripts -> Conversation."""

    def __init__(self, segmenter: UtteranceSegmenter, stt: Parakeet, convo: Conversation,
                 player: Player, barge_in: bool = False, on_event=_no_event, log=_print):
        self.segmenter = segmenter
        self.stt = stt
        self.convo = convo
        self.player = player
        self.barge_in = barge_in
        self.emit = on_event
        self.log = log
        self.muted = False
        self.guess = ""  # speculative transcript started at the early pause
        self.stt_ms = 0.0

    def _transcribe(self, audio) -> str:
        t0 = time.perf_counter()
        text = self.stt.transcribe(audio, MIC_RATE)
        self.stt_ms = (time.perf_counter() - t0) * 1000
        return text

    def handle(self, frame) -> str | None:
        """Process one mic frame. Returns 'unmuted' when the caller should drop
        stale buffered frames."""
        if not self.barge_in:
            # half-duplex: ignore the mic while the assistant talks (and a short echo tail)
            busy = self.convo.active() or time.monotonic() - self.player.last_sound < ECHO_TAIL_S
            if busy:
                if not self.muted:
                    self.segmenter.reset()
                    self.muted = True
                return None
            if self.muted:
                self.muted = False
                return "unmuted"

        event, audio = self.segmenter.push(frame)
        if event == "start":
            self.emit("user_start")
            if self.barge_in and self.convo.active():
                self.convo.interrupt()
                self.log("  [interrupted]")
                self.emit("interrupted")
        elif event == "pause" and audio is not None:
            # Likely end of turn: start replying now, play only once confirmed.
            self.guess = self._transcribe(audio)
            if self.guess:
                self.emit("user_partial", text=self.guess)
                self.convo.reply_to(self.guess, speculative=True)
        elif event == "resume" and self.guess:
            self.convo.retract()  # you kept talking
            self.guess = ""
        elif event == "end":
            if self.guess:
                self.convo.confirm()
                self.log(f"You: {self.guess}   (STT {self.stt_ms:.0f} ms, started early)")
                self.emit("user_final", text=self.guess, stt_ms=self.stt_ms)
                self.guess = ""
            elif audio is not None:
                text = self._transcribe(audio)
                if text:
                    self.log(f"You: {text}   (STT {self.stt_ms:.0f} ms)")
                    self.emit("user_final", text=text, stt_ms=self.stt_ms)
                    self.convo.reply_to(text)
                else:
                    self.emit("user_cancel")
            else:
                self.emit("user_cancel")  # too short to be speech
        return event


def check_server(base_url: str, api_key: str, wanted: list[str]) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = requests.get(base_url.rstrip("/") + "/models", headers=headers, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Can't reach the LLM server at {base_url} ({e}).\n"
                 "In Jan: Settings > Local API Server > Start Server.")
    available = {m["id"] for m in r.json().get("data", [])}
    for name in wanted:
        if name not in available:
            print(f"Warning: model {name!r} isn't listed by the server "
                  f"(available: {', '.join(sorted(available)) or 'none'}).")


def warm_up(llm: LLMClient, tts, name: str) -> None:
    """Load the models and report whether TTS is faster than real time."""
    cancel = threading.Event()
    print("Warming up the LLM...", flush=True)
    for _ in llm.stream([{"role": "user", "content": "Say hi."}], cancel):
        pass
    print(f"Warming up {name}...", flush=True)
    list(tts.synthesize("Warm up.", cancel))
    start = time.perf_counter()
    samples = sum(len(c) for c in tts.synthesize("Hello there, it's nice to meet you.", cancel))
    elapsed = time.perf_counter() - start
    rtf = (samples / TTS_RATE) / elapsed if elapsed else 0
    print(f"{name} speed: {rtf:.2f}x real time"
          + ("" if rtf >= 1 else " (slower than real time: replies will pause to buffer)"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:1337/v1", help="Jan / OpenAI-compatible API")
    p.add_argument("--api-key", default="", help="API key, if set in Jan's Local API Server")
    p.add_argument("--llm-model", default="Qwen3_5-4B-IQ4_XS")
    p.add_argument("--tts", default="kokoro", choices=["kokoro", "orpheus"],
                   help="kokoro: Kokoro-82M, fast, on the CPU; orpheus: more expressive, "
                        "slower, needs the GPU (via Jan)")
    p.add_argument("--tts-model", default="Orpheus-3b-FT-Q4_K_M", help="Jan model ID (orpheus)")
    p.add_argument("--voice", default=None,
                   help=f"kokoro: {', '.join(KOKORO_VOICES)} (default af_heart); "
                        f"orpheus: {', '.join(ORPHEUS_VOICES)} (default tara)")
    p.add_argument("--speed", type=float, default=1.0, help="speaking rate (kokoro)")
    p.add_argument("--tts-cpu", action="store_true",
                   help="run Kokoro on the CPU even when CUDA is available")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--barge-in", action="store_true",
                   help="let your voice interrupt the assistant (use headphones, or it hears itself)")
    p.add_argument("--stt-gpu", action="store_true",
                   help="run Parakeet on the GPU (fp32, ~2.5 GB extra VRAM)")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--silence-ms", type=int, default=600, help="pause length that ends your turn")
    p.add_argument("--early-ms", type=int, default=250,
                   help="pause length that starts a speculative reply (>= --silence-ms disables)")
    p.add_argument("--prebuffer", type=float, default=0.1,
                   help="extra jitter buffer in seconds (each sentence is also buffered "
                        "automatically based on measured Orpheus speed)")
    p.add_argument("--tts-parallel", type=int, default=2,
                   help="sentences synthesized concurrently (Jan batches them)")
    p.add_argument("--input-device", default=None, help="mic name or index (see --list-devices)")
    p.add_argument("--output-device", default=None, help="speaker name or index")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()
    for attr in ("input_device", "output_device"):
        v = getattr(args, attr)
        if v is not None and v.isdigit():
            setattr(args, attr, int(v))
    return args


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    orpheus = args.tts == "orpheus"
    check_server(args.base_url, args.api_key, [args.llm_model] + ([args.tts_model] if orpheus else []))

    print("Loading Silero VAD and Parakeet TDT 0.6B v2...", flush=True)
    segmenter = UtteranceSegmenter(SileroVAD(ensure_silero(MODELS_DIR)),
                                   threshold=args.vad_threshold, min_silence_ms=args.silence_ms,
                                   early_silence_ms=args.early_ms)
    stt = Parakeet(gpu=args.stt_gpu)
    llm = LLMClient(args.base_url, args.llm_model, args.api_key)
    if orpheus:
        tts = OrpheusTTS(args.base_url, args.tts_model, voice=args.voice or "tara",
                         api_key=args.api_key)
    else:
        cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        tts = KokoroTTS(MODELS_DIR, voice=args.voice or "af_heart", speed=args.speed,
                        gpu=cuda and not args.tts_cpu)
    if not args.no_warmup:
        warm_up(llm, tts, args.tts.capitalize())

    player = Player(TTS_RATE, prebuffer_s=args.prebuffer, device=args.output_device)
    convo = Conversation(llm, tts, player, args.system_prompt or system_prompt(sound_tags=orpheus),
                         tts_parallel=args.tts_parallel)

    mode = "barge-in on" if args.barge_in else "half-duplex: wait for the reply to finish"
    print(f"\nListening ({mode}). Speak whenever you're ready. Ctrl+C to quit.\n", flush=True)

    listener = Listener(segmenter, stt, convo, player, barge_in=args.barge_in)
    try:
        with Microphone(MIC_RATE, FRAME, device=args.input_device) as mic:
            while True:
                if listener.handle(mic.frames.get()) == "unmuted":
                    while not mic.frames.empty():  # drop audio captured while muted
                        mic.frames.get_nowait()
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        convo.interrupt()
        player.close()


if __name__ == "__main__":
    main()
