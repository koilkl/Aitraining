"""
Desktop launcher using cefpython3 for Windows
This provides a native Chromium window that's more stable than pywebview on Windows
"""
from __future__ import annotations

import contextlib
import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


# ================================================
# Monkey patch importlib.metadata for streamlit
# ================================================
def monkey_patch_metadata():
    try:
        import importlib.metadata
        ver = importlib.metadata.version('streamlit')
        return
    except Exception:
        pass
    
    try:
        import sys
        from importlib.metadata import Distribution, PackageNotFoundError
        import importlib.metadata
        
        try:
            import streamlit
            if hasattr(streamlit, '__version__'):
                STREAMLIT_VERSION = streamlit.__version__
            else:
                STREAMLIT_VERSION = '1.28.0'
        except Exception:
            STREAMLIT_VERSION = '1.28.0'
        
        class FakeDistribution(Distribution):
            def __init__(self, name, version):
                self._name = name
                self._version = version
            
            @property
            def metadata(self):
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
        
        original_from_name = importlib.metadata.Distribution.from_name
        
        def patched_from_name(name):
            if name == 'streamlit':
                return FakeDistribution(name, STREAMLIT_VERSION)
            return original_from_name(name)
        
        importlib.metadata.Distribution.from_name = staticmethod(patched_from_name)
        
        original_version = importlib.metadata.version
        
        def patched_version(name):
            if name == 'streamlit':
                return STREAMLIT_VERSION
            return original_version(name)
        
        importlib.metadata.version = patched_version
        
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


def _run_streamlit_server(port: int, log_path: str) -> None:
    import traceback

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
    with log_file.open("a", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                bootstrap.load_config_options(flag_options=flag_options)
                bootstrap.run(str(app_py), False, [], flag_options)
            except Exception:
                f.write(traceback.format_exc())
                f.write("\n")
                f.flush()
                raise


def _run_cef_window(url: str, title: str = "TFLite Training") -> None:
    """
    Use cefpython3 to create a native Chromium window
    cefpython3 is more stable on Windows than pywebview
    """
    try:
        from cefpython3.cefpython import cefpython
    except ImportError:
        print("cefpython3 not found, falling back to browser")
        webbrowser.open(url)
        return
    
    # Configure cefpython
    cef_settings = {
        "multi_threaded_message_loop": True,
        "context_menu": {"enabled": True},
        "remote_debugging_port": -1,
    }
    
    browser_settings = {
        "windowless_rendering_enabled": False,
        "javascript_close_windows_disallowed": False,
    }
    
    # Initialize cefpython
    cefpython.Initialize(settings=cef_settings)
    
    # Create window
    window_info = cefpython.WindowInfo()
    window_info.SetAsPopup(False, title)
    
    # Create browser
    browser = cefpython.CreateBrowserSync(
        window_info=window_info,
        url=url,
        settings=browser_settings
    )
    
    # Set window size
    browser.SetOuterSize(1200, 800)
    browser.SetWindowTitle(title)
    
    # Message loop
    cefpython.MessageLoop()
    
    # Shutdown
    cefpython.Shutdown()


def main() -> None:
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_file = (_app_data_dir() / "logs" / "streamlit.log").resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    log_path = str(log_file)
    
    print(f"Starting Streamlit server on {url}...")
    
    # Start streamlit in a daemon thread
    server_thread = threading.Thread(target=_run_streamlit_server, args=(port, log_path), daemon=True)
    server_thread.start()
    
    try:
        print("Waiting for server to be ready...")
        _wait_http_ready(url)
        
        print("Opening desktop window...")
        
        # Try cefpython3 first, fallback to browser
        if sys.platform == "win32":
            try:
                _run_cef_window(url)
            except Exception as e:
                print(f"cefpython3 failed: {e}, falling back to browser")
                webbrowser.open(url)
        else:
            webbrowser.open(url)
        
        print("App is running! Close the window to stop.")
        print("Press Ctrl+C to exit.")
        
        while server_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pass


if __name__ == "__main__":
    main()
