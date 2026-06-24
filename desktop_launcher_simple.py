from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


# ================================================
# FIX: Monkey patch importlib.metadata for streamlit
# When packaged with PyInstaller, metadata is missing
# ================================================
def monkey_patch_metadata():
    # First, try to get real streamlit version from package
    try:
        import importlib.metadata
        # If it works, great!
        ver = importlib.metadata.version('streamlit')
        return
    except Exception:
        # If metadata fails, let's create our own metadata
        pass
    
    # Create fake metadata
    try:
        import sys
        from importlib.metadata import Distribution, PackageNotFoundError
        import importlib.metadata
        
        # Try to find streamlit's actual version another way
        try:
            import streamlit
            if hasattr(streamlit, '__version__'):
                STREAMLIT_VERSION = streamlit.__version__
            else:
                # Common known versions if we can't find it
                STREAMLIT_VERSION = '1.28.0'
        except Exception:
            STREAMLIT_VERSION = '1.28.0'
        
        # Create a fake distribution class
        class FakeDistribution(Distribution):
            def __init__(self, name, version):
                self._name = name
                self._version = version
            
            @property
            def metadata(self):
                # Return minimal metadata
                class FakeMetadata:
                    def __getitem__(self, key):
                        if key == 'Name':
                            return self._name
                        if key == 'Version':
                            return self._version
                        return ''
                    def get(self, key, default=None):
                        try:
                            return self[key]
                        except:
                            return default
                return FakeMetadata()
            
            @property
            def name(self):
                return self._name
            
            @property
            def version(self):
                return self._version
        
        # Monkey patch the from_name function
        original_from_name = importlib.metadata.Distribution.from_name
        
        def patched_from_name(name):
            if name == 'streamlit':
                return FakeDistribution(name, STREAMLIT_VERSION)
            return original_from_name(name)
        
        importlib.metadata.Distribution.from_name = staticmethod(patched_from_name)
        
        # Also monkey patch the version function
        original_version = importlib.metadata.version
        
        def patched_version(name):
            if name == 'streamlit':
                return STREAMLIT_VERSION
            return original_version(name)
        
        importlib.metadata.version = patched_version
        
        # Also monkey patch distribution
        original_distribution = importlib.metadata.distribution
        
        def patched_distribution(name):
            if name == 'streamlit':
                return FakeDistribution(name, STREAMLIT_VERSION)
            return original_distribution(name)
        
        importlib.metadata.distribution = patched_distribution
        
        print(f"Monkey-patched streamlit metadata: version {STREAMLIT_VERSION}")
        
    except Exception as e:
        print(f"Metadata patch failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the entire app, continue anyway


# ================================================
# Original code
# ================================================

def _resource_path(rel_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return (Path(getattr(sys, "_MEIPASS")) / rel_path).resolve()
    return (Path(__file__).resolve().parent / rel_path).resolve()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ready(url: str, timeout_s: float = 25.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception as e:
            last_err = e
        time.sleep(0.25)
    raise RuntimeError(f"Streamlit not ready: {url} ({last_err})")


def _app_data_dir() -> Path:
    env_override = os.getenv("TFLITE_TRAINING_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (base / "TFLiteTraining").resolve()


def _find_app_browser() -> str | None:
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    if sys.platform == "win32":
        candidates.extend(
            [
                os.path.join(program_files_x86, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(program_files, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(local_app_data, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(program_files_x86, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe"),
            ]
        )
    for name in ("msedge.exe", "chrome.exe"):
        found = shutil.which(name)
        if found:
            candidates.insert(0, found)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _launch_app_window(url: str) -> subprocess.Popen | None:
    browser = _find_app_browser()
    if not browser:
        webbrowser.open(url)
        return None
    args = [
        browser,
        f"--app={url}",
        "--new-window",
        "--disable-session-crashed-bubble",
        "--disable-features=Translate,msWebRTCCdpInterceptor",
        "--window-size=1280,900",
    ]
    return subprocess.Popen(args)


def _run_streamlit_server(port: int, log_path: str) -> None:
    import traceback

    # Apply the metadata patch before importing streamlit
    monkey_patch_metadata()

    from streamlit.web import bootstrap

    app_py = _resource_path("app.py")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")

    flag_options = {
        "global_developmentMode": False,
        "server_headless": True,
        "server_port": port,
        "server_address": "127.0.0.1",
        "browser_gatherUsageStats": False,
        "browser_serverPort": port,
        "browser_serverAddress": "127.0.0.1",
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        try:
            f.write(f"[server] start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.flush()
        except Exception:
            pass
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                bootstrap.load_config_options(flag_options=flag_options)
                bootstrap.run(str(app_py), False, [], flag_options)
            except Exception:
                f.write(traceback.format_exc())
                f.write("\n")
                f.flush()
                raise


def _server_command(port: int, log_path: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--serve", f"--port={int(port)}", f"--log={log_path}"]
    return [sys.executable, str(Path(__file__).resolve()), "--serve", f"--port={int(port)}", f"--log={log_path}"]


def _launch_server_process(port: int, log_path: str) -> subprocess.Popen:
    cmd = _server_command(port, log_path)
    return subprocess.Popen(cmd)


def _maybe_run_server_mode() -> bool:
    if "--serve" not in sys.argv:
        return False
    port = None
    log_path = None
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--log="):
            log_path = arg.split("=", 1)[1]
    if port is None or not log_path:
        raise RuntimeError("missing --port or --log for serve mode")
    _run_streamlit_server(port, log_path)
    return True


def main() -> None:
    if _maybe_run_server_mode():
        return

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_file = (_app_data_dir() / "logs" / "streamlit.log").resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    log_path = str(log_file)
    
    print(f"Starting Streamlit server on {url}...")
    server_proc = _launch_server_process(port, log_path)
    
    try:
        print("Waiting for server to be ready...")
        _wait_http_ready(url)
        
        print("Opening app window...")
        browser_proc = _launch_app_window(url)
        
        print("App is running! Close this window to stop.")
        print("Press Ctrl+C to exit.")
        
        while True:
            if server_proc.poll() is not None:
                raise RuntimeError(f"Streamlit server exited unexpectedly. Log: {log_path}")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except Exception:
                server_proc.kill()


if __name__ == "__main__":
    main()
