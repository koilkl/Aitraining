"""
Desktop launcher - macOS style using pywebview
Fixed for Windows to avoid recursion depth issues
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.request
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


def _debug_post(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    env_path = Path(".dbg/open-project-layout.env")
    url = "http://127.0.0.1:7777/event"
    session_id = "open-project-layout"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    url = line.split("=", 1)[1].strip() or url
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip() or session_id
    except Exception:
        pass
    payload = {
        "sessionId": session_id,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=1.5).read()
    except Exception:
        pass


def _run_streamlit_server(port: int, log_path: str) -> None:
    import traceback

    monkey_patch_metadata()

    from streamlit.web import bootstrap

    app_py = _resource_path("app.py")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")

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


_LAYOUT_REFRESH_JS = r"""
(() => {
  const refreshTarget = (win) => {
    if (!win) return;
    try { win.dispatchEvent(new Event('resize')); } catch (e) {}
    try { win.dispatchEvent(new Event('orientationchange')); } catch (e) {}
  };
  refreshTarget(window);
  try {
    document.querySelectorAll('iframe').forEach((frame) => {
      try { refreshTarget(frame.contentWindow); } catch (e) {}
    });
  } catch (e) {}
  return true;
})();
"""


def _schedule_window_layout_refresh(window: "webview.Window", reason: str = "") -> None:
    def _run_once(delay_s: float) -> None:
        def _inner() -> None:
            try:
                # Use evaluate_js but catch any errors
                result = None
                try:
                    result = window.evaluate_js(_LAYOUT_REFRESH_JS)
                except Exception:
                    pass
            except Exception:
                pass

        timer = threading.Timer(delay_s, _inner)
        timer.daemon = True
        timer.start()

    for delay_s in (0.0, 0.12, 0.35, 0.8):
        _run_once(delay_s)


class _ShellApi:
    def __init__(self) -> None:
        self.window = None

    def bind(self, window: "webview.Window") -> None:
        self.window = window

    def request_reflow(self, reason: str = "") -> bool:
        if self.window is None:
            return False
        _schedule_window_layout_refresh(self.window, reason=reason)
        return True


def _startup_window_logic(window: "webview.Window") -> None:
    _schedule_window_layout_refresh(window, reason="startup")


def main() -> None:
    multiprocessing.freeze_support()
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_file = (_app_data_dir() / "logs" / "streamlit.log").resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    log_path = str(log_file)
    proc = multiprocessing.Process(target=_run_streamlit_server, args=(port, log_path), daemon=True)
    proc.start()
    try:
        deadline = time.time() + 25.0
        last_err: Exception | None = None
        while time.time() < deadline:
            if not proc.is_alive():
                break
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if 200 <= resp.status < 500:
                        last_err = None
                        break
            except Exception as e:
                last_err = e
            time.sleep(0.25)

        if last_err is not None:
            raise RuntimeError(f"Streamlit not ready: {url} ({last_err}). Log: {log_path}")
        if not proc.is_alive():
            raise RuntimeError(f"Streamlit process exited. Log: {log_path}")

        import webview

        # On Windows, try to avoid recursion issues with accessibility APIs
        if sys.platform == "win32":
            os.environ['PYWEBVIEW_GUI'] = 'edgechromium'
            os.environ['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = '--disable-gpu --disable-gpu-compositing --disable-features=msWebRTCCdpInterceptor'

        shell_api = _ShellApi()
        window = webview.create_window("TF Lite Training", url, width=1200, height=800, js_api=shell_api)
        shell_api.bind(window)
        
        # Bind events with error handling to avoid recursion
        try:
            window.events.loaded += lambda: _schedule_window_layout_refresh(window, reason="loaded")
        except Exception:
            pass
        
        try:
            window.events.shown += lambda: _schedule_window_layout_refresh(window, reason="shown")
        except Exception:
            pass
        
        try:
            window.events.closed += lambda: _shutdown_and_exit(proc)
        except Exception:
            pass
        
        webview.start(_startup_window_logic, window, debug=(sys.platform == "win32"))
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def _shutdown_and_exit(proc: multiprocessing.Process) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    os._exit(0)


if __name__ == "__main__":
    main()
