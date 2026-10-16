"""Jan integration: find its folders, install GGUFs into it, limit loaded models, restart it."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import requests

# Models the UI can install into Jan. IDs match the folder names Jan uses.
CATALOG = [
    {"id": "Qwen3_5-4B-IQ4_XS", "kind": "llm", "label": "Qwen 3.5 4B IQ4_XS",
     "repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-IQ4_XS.gguf"},
    {"id": "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M", "kind": "llm",
     "label": "Gemma 4 E2B IQ3_M", "repo": "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
     "file": "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"},
    {"id": "Orpheus-3b-FT-Q4_K_M", "kind": "tts", "label": "Orpheus 3B Q4_K_M",
     "repo": "lex-au/Orpheus-3b-FT-Q4_K_M.gguf", "file": "Orpheus-3b-FT-Q4_K_M.gguf"},
]


def app_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Jan"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Jan"
    return Path.home() / ".config" / "Jan"


def data_dir() -> Path:
    try:
        return Path(json.loads((app_dir() / "settings.json").read_text())["data_folder"])
    except (OSError, KeyError, ValueError):
        return app_dir() / "data"


def models_dir() -> Path:
    return data_dir() / "llamacpp" / "models"


def installed(model_id: str) -> bool:
    return (models_dir() / model_id / "model.gguf").exists()


def remote_info(repo: str, file: str) -> tuple[int, str | None]:
    """(size, sha256) from the Hugging Face API."""
    r = requests.get(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=30)
    r.raise_for_status()
    for f in r.json():
        if f["path"] == file:
            return f["size"], (f.get("lfs") or {}).get("oid")
    raise FileNotFoundError(f"{file} not found in {repo}")


def install(entry: dict, progress: Callable[[int, int], None]) -> None:
    """Download a catalog GGUF into Jan's model folder (resumable, sha256-checked).
    Jan only scans for models at startup, so restart it afterwards."""
    size, sha = remote_info(entry["repo"], entry["file"])
    folder = models_dir() / entry["id"]
    folder.mkdir(parents=True, exist_ok=True)
    final, part = folder / "model.gguf", folder / "model.gguf.part"
    url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['file']}"
    for attempt in range(30):  # flaky networks: resume where we left off
        done = part.stat().st_size if part.exists() else 0
        if done >= size:
            break
        try:
            with requests.get(url, headers={"Range": f"bytes={done}-"}, stream=True,
                              timeout=60) as r:
                r.raise_for_status()
                with part.open("ab") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        done += len(block)
                        progress(done, size)
        except requests.RequestException:
            time.sleep(3)
    if part.stat().st_size < size:
        raise RuntimeError("download kept failing; try again later")
    if sha:
        h = hashlib.sha256()
        with part.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != sha:
            part.unlink()
            raise RuntimeError("checksum mismatch; the download was corrupted, please retry")
    part.replace(final)
    (folder / "model.yml").write_text(
        f"model_path: llamacpp/models/{entry['id']}/model.gguf\nname: {entry['id']}\n"
        f"size_bytes: {size}\n", encoding="utf-8")


# ---------------------------------------------------------------- loaded-model limit

def _settings_path() -> Path:
    return data_dir() / "llamacpp" / "settings.json"


def get_models_max() -> int | None:
    """Jan's "max loaded models" (0 = unlimited, which keeps every model in VRAM)."""
    try:
        for s in json.loads(_settings_path().read_text(encoding="utf-8")):
            if s.get("key") == "models_max":
                return int(s["controllerProps"]["value"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def set_models_max(n: int) -> None:
    """Only call while Jan is closed: it rewrites this file on exit."""
    path = _settings_path()
    settings = json.loads(path.read_text(encoding="utf-8"))
    for s in settings:
        if s.get("key") == "models_max":
            s["controllerProps"]["value"] = n
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- restart

def _exe() -> Path | None:
    if sys.platform != "win32":
        return None
    for name in ("Jan-Desktop.exe", "Jan.exe"):
        p = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Jan" / name
        if p.exists():
            return p
    return None


def is_installed() -> bool:
    return _exe() is not None


def is_running() -> bool:
    if sys.platform != "win32":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Jan-Desktop.exe"], capture_output=True,
                         text=True).stdout
    return "Jan-Desktop.exe" in out


def start() -> None:
    exe = _exe()
    if exe:
        subprocess.Popen([str(exe)], creationflags=subprocess.DETACHED_PROCESS
                         | subprocess.CREATE_NEW_PROCESS_GROUP, close_fds=True)


def wait_for_api(base_url: str, timeout: float = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(base_url.rstrip("/") + "/models", timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def ensure_api_autostart() -> bool:
    """Turn on Settings → Local API Server → start on launch. Only while Jan is closed.
    Returns False if Jan hasn't created its settings yet (first launch does that)."""
    path = app_dir() / "data" / "settings.json"
    path = data_dir() / "settings.json" if not path.exists() else path
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        server = json.loads(settings["setting-local-api-server"])
    except (OSError, KeyError, ValueError):
        return False
    if not server.get("state", {}).get("enableOnStartup"):
        server.setdefault("state", {})["enableOnStartup"] = True
        settings["setting-local-api-server"] = json.dumps(server)
        path.write_text(json.dumps(settings), encoding="utf-8")
    return True


def restart(base_url: str, models_max: int | None = None, timeout: float = 90) -> None:
    """Quit Jan (freeing all VRAM), optionally change its loaded-model limit, start it
    again and wait for the API server (Jan starts it on launch when enabled)."""
    exe = _exe()
    if exe is None:
        raise RuntimeError("Restarting Jan is only supported on Windows with Jan installed")
    for image in ("Jan-Desktop.exe", "Jan.exe", "jan-llama-worker.exe", "llama-server.exe"):
        subprocess.run(["taskkill", "/F", "/T", "/IM", image], capture_output=True)
    time.sleep(1.5)
    if models_max is not None:
        set_models_max(models_max)
    ensure_api_autostart()
    start()
    if not wait_for_api(base_url, timeout):
        raise RuntimeError("Jan restarted, but its API server didn't come up. "
                           "Enable Settings → Local API Server → start automatically.")
