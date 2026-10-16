"""Benchmark each pipeline stage and the end-to-end voice latency.

    uv run python bench/bench.py stt vad snac            # offline parts, no server needed
    uv run python bench/bench.py llm tts e2e --llm-model Qwen3_5-4B-IQ4_XS

Test utterances: any `name.wav` (16 kHz mono) with a matching `name.txt` transcript
in --speech-dir.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

from realtime_chat.llm import SYSTEM_PROMPT, LLMClient, sentences
from realtime_chat.main import MODELS_DIR
from realtime_chat.stt import Parakeet
from realtime_chat.tts import SAMPLE_RATE as TTS_RATE
from realtime_chat.tts import WINDOW, OrpheusTTS, SnacDecoder
from realtime_chat.vad import FRAME, SAMPLE_RATE, SileroVAD, UtteranceSegmenter, ensure_silero

PROMPTS = [
    "What's a good way to stay focused while studying?",
    "Explain what a black hole is in simple terms.",
    "Tell me a short joke about computers.",
    "I'm feeling a bit stressed about exams. Any advice?",
]
TTS_SENTENCES = [
    "Hey there! It's really nice to finally talk to you.",
    "The quick brown fox jumps over the lazy dog, and then it takes a nap in the sun.",
    "Honestly, I think the best way to learn is to build something you actually care about.",
]


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    n = int(len(audio) * dst / src)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def wer(ref: str, hyp: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def vram_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def ms(xs: list[float]) -> str:
    return f"{statistics.median(xs) * 1000:.0f} ms (median of {len(xs)})"


# --------------------------------------------------------------------------- stages

def bench_stt(args, utterances) -> dict:
    res = {}
    for label, gpu in [("cpu-int8", False)] + ([("gpu-fp32", True)] if args.stt_gpu else []):
        stt = Parakeet(gpu=gpu)
        stt.transcribe(utterances[0][1])  # warm-up
        times, errs = [], []
        for name, audio, ref in utterances:
            t = time.perf_counter()
            hyp = stt.transcribe(audio)
            times.append(time.perf_counter() - t)
            errs.append(wer(ref, hyp))
            print(f"  [{label}] {name}: {times[-1]*1000:.0f} ms  WER {errs[-1]:.0%}  -> {hyp}")
        res[label] = {"latency": ms(times), "wer": f"{statistics.mean(errs):.1%}",
                      "audio_s": round(statistics.mean(len(a) / SAMPLE_RATE for _, a, _ in utterances), 1)}
    return res


def bench_vad(args, utterances) -> dict:
    seg = UtteranceSegmenter(SileroVAD(ensure_silero(MODELS_DIR)), min_silence_ms=args.silence_ms)
    step, ends = [], []
    for _, audio, _ in utterances:
        seg.reset()
        stream = np.concatenate([np.zeros(SAMPLE_RATE // 2, np.float32), audio,
                                 np.zeros(SAMPLE_RATE * 2, np.float32)])
        loud = np.flatnonzero(np.abs(audio) > 0.01)  # recordings may carry trailing silence
        speech_end = (SAMPLE_RATE // 2 + (loud[-1] if len(loud) else len(audio))) / SAMPLE_RATE
        for i in range(0, len(stream) - FRAME + 1, FRAME):
            t = time.perf_counter()
            event, _ = seg.push(stream[i : i + FRAME])
            step.append(time.perf_counter() - t)
            if event == "end":
                ends.append((i + FRAME) / SAMPLE_RATE - speech_end)
                break
    return {"per_frame": f"{statistics.median(step) * 1e3:.2f} ms per 32 ms frame",
            "end_of_turn_delay": f"{statistics.median(ends) * 1000:.0f} ms after speech stops"}


def bench_snac(args) -> dict:
    dec = SnacDecoder()
    rng = np.random.default_rng(0)
    res = {}
    for hop in (1, 2, 4, 8):
        codes = rng.integers(1, 4096, (hop + 3) * 7).tolist()  # hop frames + 3 of context
        dec.decode(codes)
        times = []
        for _ in range(20):
            t = time.perf_counter()
            dec.decode(codes)
            times.append(time.perf_counter() - t)
        budget = hop * 2048 / TTS_RATE
        res[f"hop{hop}"] = (f"{statistics.median(times)*1000:.0f} ms per {budget*1000:.0f} ms of audio "
                            f"= {statistics.median(times) / budget:.0%} CPU of real time")
    return res


def bench_llm(args) -> dict:
    llm = LLMClient(args.base_url, args.llm_model, max_tokens=200)
    cancel = threading.Event()
    for _ in llm.stream([{"role": "user", "content": "hi"}], cancel):
        pass  # load model
    ttft, tfs, rates, replies = [], [], [], []
    for p in PROMPTS:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
        t0 = time.perf_counter()
        first_tok = first_sent = None
        chars = 0
        sents = []

        def deltas():
            nonlocal first_tok, chars
            for d in llm.stream(msgs, cancel):
                if first_tok is None:
                    first_tok = time.perf_counter() - t0
                chars += len(d)
                yield d

        for s in sentences(deltas()):
            if first_sent is None:
                first_sent = time.perf_counter() - t0
            sents.append(s)
        total = time.perf_counter() - t0
        ttft.append(first_tok)
        tfs.append(first_sent)
        rates.append((chars / 4) / max(total - first_tok, 1e-3))  # ~4 chars per token
        replies.append(" ".join(sents))
        print(f"  {p}\n    -> {replies[-1]}")
    return {"model": args.llm_model, "first_token": ms(ttft), "first_sentence": ms(tfs),
            "speed": f"~{statistics.median(rates):.0f} tok/s", "replies": replies}


def bench_tts(args, stt: Parakeet | None, load_llm: bool) -> dict:
    tts = OrpheusTTS(args.base_url, args.tts_model, voice=args.voice)
    cancel = threading.Event()
    list(tts.synthesize("Warming up.", cancel))
    res = {}
    modes = [("alone", False)] + ([("while LLM generates", True)] if load_llm else [])
    for label, contend in modes:
        first, rtfs, errs = [], [], []
        stop = threading.Event()
        if contend:
            llm = LLMClient(args.base_url, args.llm_model, max_tokens=2000)

            def hog():
                while not stop.is_set():
                    for _ in llm.stream([{"role": "user", "content": "Write a long story."}], stop):
                        pass
            threading.Thread(target=hog, daemon=True).start()
            time.sleep(1.5)
        for text in TTS_SENTENCES:
            t0 = time.perf_counter()
            chunks, t_first = [], None
            for c in tts.synthesize(text, cancel):
                t_first = t_first or time.perf_counter() - t0
                chunks.append(c)
            wall = time.perf_counter() - t0
            audio = np.concatenate(chunks)
            first.append(t_first)
            rtfs.append(len(audio) / TTS_RATE / wall)
            if stt:
                errs.append(wer(text, stt.transcribe(resample(audio, TTS_RATE, SAMPLE_RATE))))
            if args.save_audio:
                save_wav(Path(args.save_audio) / f"{label[:5]}_{len(rtfs)}.wav", audio, TTS_RATE)
        stop.set()
        res[label] = {"first_audio": ms(first), "speed": f"{statistics.median(rtfs):.2f}x real time",
                      **({"intelligibility_wer": f"{statistics.mean(errs):.1%}"} if errs else {})}
        print(f"  [{label}] {res[label]}")
    return res


def bench_e2e(args, utterances, stt: Parakeet) -> dict:
    """Silence after you stop talking -> first audio sample, minus playback prebuffer."""
    seg = UtteranceSegmenter(SileroVAD(ensure_silero(MODELS_DIR)), min_silence_ms=args.silence_ms)
    llm = LLMClient(args.base_url, args.llm_model)
    tts = OrpheusTTS(args.base_url, args.tts_model, voice=args.voice)
    cancel = threading.Event()
    totals, parts = [], []
    for _, audio, _ in utterances[:3]:
        seg.reset()
        stream = np.concatenate([audio, np.zeros(SAMPLE_RATE * 2, np.float32)])
        for i in range(0, len(stream) - FRAME + 1, FRAME):
            event, utt = seg.push(stream[i : i + FRAME])
            if event == "end":
                loud = np.flatnonzero(np.abs(audio) > 0.01)
                vad_delay = (i + FRAME - (loud[-1] if len(loud) else len(audio))) / SAMPLE_RATE
                break
        t0 = time.perf_counter()
        text = stt.transcribe(utt)
        t_stt = time.perf_counter() - t0
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        first_sentence = next(sentences(llm.stream(msgs, cancel)))
        t_llm = time.perf_counter() - t0 - t_stt
        next(tts.synthesize(first_sentence, cancel))
        t_tts = time.perf_counter() - t0 - t_stt - t_llm
        total = vad_delay + t_stt + t_llm + t_tts + args.prebuffer
        totals.append(total)
        parts.append((vad_delay, t_stt, t_llm, t_tts))
        print(f"  {text!r}: VAD {vad_delay*1000:.0f} + STT {t_stt*1000:.0f} + LLM 1st sentence "
              f"{t_llm*1000:.0f} + TTS 1st audio {t_tts*1000:.0f} + prebuffer {args.prebuffer*1000:.0f}"
              f" = {total*1000:.0f} ms")
    med = lambda k: f"{statistics.median(p[k] for p in parts) * 1000:.0f} ms"
    return {"voice_to_voice": ms(totals), "vad": med(0), "stt": med(1), "llm_first_sentence": med(2),
            "tts_first_audio": med(3), "prebuffer": f"{args.prebuffer*1000:.0f} ms"}


def save_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stages", nargs="+", choices=["stt", "vad", "snac", "llm", "tts", "e2e"])
    p.add_argument("--speech-dir", type=Path, required=False)
    p.add_argument("--base-url", default="http://127.0.0.1:1337/v1")
    p.add_argument("--llm-model", default="Qwen3_5-4B-IQ4_XS")
    p.add_argument("--tts-model", default="Orpheus-3b-FT-Q4_K_M")
    p.add_argument("--voice", default="tara")
    p.add_argument("--stt-gpu", action="store_true", help="also benchmark Parakeet on the GPU")
    p.add_argument("--silence-ms", type=int, default=600)
    p.add_argument("--prebuffer", type=float, default=0.3)
    p.add_argument("--save-audio", default=None, help="folder to save Orpheus samples")
    p.add_argument("--no-contention", action="store_true", help="skip TTS-while-LLM-runs test")
    p.add_argument("--out", type=Path, default=None, help="append results as JSON lines")
    args = p.parse_args()

    utterances = []
    if args.speech_dir:
        for wav in sorted(args.speech_dir.glob("*.wav")):
            utterances.append((wav.stem, load_wav(wav), wav.with_suffix(".txt").read_text().strip()))

    results = {"vram_idle_mb": vram_mb()}
    stt = None
    if "stt" in args.stages:
        results["stt"] = bench_stt(args, utterances)
    if "vad" in args.stages:
        results["vad"] = bench_vad(args, utterances)
    if "snac" in args.stages:
        results["snac"] = bench_snac(args)
    if "llm" in args.stages:
        results["llm"] = bench_llm(args)
    if {"tts", "e2e"} & set(args.stages):
        try:
            stt = Parakeet()
        except Exception as e:  # e.g. weights not downloaded yet
            print(f"  Parakeet unavailable, skipping intelligibility: {e.__class__.__name__}")
    if "tts" in args.stages:
        results["tts"] = bench_tts(args, stt, load_llm=not args.no_contention)
    if "e2e" in args.stages:
        results["e2e"] = bench_e2e(args, utterances, stt)
    results["vram_after_mb"] = vram_mb()

    print(json.dumps({k: v for k, v in results.items()}, indent=2))
    if args.out:
        with args.out.open("a") as f:
            f.write(json.dumps({"llm_model": args.llm_model, "stages": args.stages, **results}) + "\n")


if __name__ == "__main__":
    main()
