"""One-time setup: Jan, models, settings. `uv run realtime-chat-setup [--orpheus] [--gemma]`"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

from realtime_chat import jan
from realtime_chat.kokoro_tts import ensure_file
from realtime_chat.main import MODELS_DIR
from realtime_chat.stt import Parakeet
from realtime_chat.vad import ensure_silero

BASE_URL = "http://127.0.0.1:1337/v1"


def step(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def progress_printer(label: str):
    last = [0.0]

    def show(done: int, total: int) -> None:
        if time.monotonic() - last[0] > 1 or done >= total:
            last[0] = time.monotonic()
            print(f"\r   {label}: {100 * done / total:5.1f}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB",
                  end="", flush=True)
    return show


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen", action="store_true", help="also install Qwen 3.5 4B into Jan")
    p.add_argument("--no-orpheus", action="store_true",
                   help="skip Orpheus (2.4 GB); Kokoro is used for speech instead")
    args = p.parse_args()

    step("Jan")
    if not jan.is_installed():
        if sys.platform == "win32" and shutil.which("winget"):
            print("   Installing Jan with winget…")
            subprocess.run(["winget", "install", "-e", "--id", "Jan.Jan", "--accept-package-agreements",
                            "--accept-source-agreements"], check=False)
        if not jan.is_installed():
            sys.exit("   Install Jan from https://jan.ai, then run this again.")
    if not (jan.app_dir() / "settings.json").exists():
        print("   First launch of Jan to create its settings…")
        jan.start()
        jan.wait_for_api(BASE_URL, 60)
        time.sleep(5)
    running = jan.is_running()
    if running:
        print("   Jan is running; its settings are left as they are.")
        if jan.get_models_max() == 0:
            print("   Tip: Jan keeps unlimited models loaded. Set 'max loaded models' to 2 in the web UI (section 05).")
    else:
        if jan.ensure_api_autostart():
            print("   API server: starts with Jan")
        if jan.get_models_max() == 0:
            jan.set_models_max(2)
            print("   Max loaded models: 2 (frees VRAM when switching)")

    step("Models for Jan")
    wanted = ["Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M"] \
        + ([] if args.no_orpheus else ["Orpheus-3b-FT-Q4_K_M"]) \
        + (["Qwen3_5-4B-IQ4_XS"] if args.qwen else [])
    new_models = False
    for entry in (e for e in jan.CATALOG if e["id"] in wanted):
        if jan.installed(entry["id"]):
            print(f"   {entry['label']}: already installed")
            continue
        jan.install(entry, progress_printer(entry["label"]))
        print(f"\n   {entry['label']}: installed (sha256 verified)")
        new_models = True

    step("Speech models (cached under ~/.cache/huggingface)")
    ensure_silero(MODELS_DIR)
    print("   Silero VAD: ok")
    Parakeet()
    print("   Parakeet TDT 0.6B v2 (int8): ok")
    ensure_file(MODELS_DIR, "kokoro-v1.0.onnx")
    ensure_file(MODELS_DIR, "voices-v1.0.bin")
    print("   Kokoro-82M: ok")

    step("Starting Jan")
    if running and new_models:
        print("   Restarting Jan so it picks up the new models…")
        jan.restart(BASE_URL)
    elif not running:
        jan.start()
    print("   Jan API: " + ("ready" if jan.wait_for_api(BASE_URL) else
                           "not responding. In Jan: Settings → Local API Server → Start"))
    print("\nSetup complete. Run:  uv run realtime-chat-web")


if __name__ == "__main__":
    main()
