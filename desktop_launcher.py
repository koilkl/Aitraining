from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


def _resource_path(rel_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return (Path(getattr(sys, "_MEIPASS")) / rel_path).resolve()
    return (Path(__file__).resolve().parent / rel_path).resolve()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _startup_log(msg: str) -> None:
    """Append a line to a startup log so we can see where the main process hangs."""
    try:
        p = Path(tempfile.gettempdir()) / "TFLiteTraining" / "startup.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f} {msg}\n")
            f.flush()
    except Exception:
        pass


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


def _prepare_log_file() -> str:
    """Return a writable log path; fall back to the temp dir if the app-data dir fails."""
    candidates = []
    try:
        candidates.append(str((_app_data_dir() / "logs" / "streamlit.log").resolve()))
    except Exception:
        pass
    candidates.append(str((Path(tempfile.gettempdir()) / "TFLiteTraining" / "logs" / "streamlit.log").resolve()))
    for log_path in candidates:
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
            return str(p)
        except Exception:
            continue
    return ""


def _raise_fd_limit() -> None:
    target = 4096
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired_soft = max(soft, target)
        desired_hard = max(hard, desired_soft)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired_soft, desired_hard))
        except Exception:
            new_soft = min(desired_soft, hard)
            if new_soft != soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        return
    except Exception:
        pass

    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        RLIMIT_NOFILE = 8

        class Rlimit(ctypes.Structure):
            _fields_ = [("rlim_cur", ctypes.c_uint64), ("rlim_max", ctypes.c_uint64)]

        rlim = Rlimit(0, 0)
        if libc.getrlimit(RLIMIT_NOFILE, ctypes.byref(rlim)) != 0:
            return
        rlim.rlim_cur = min(max(int(rlim.rlim_cur), target), int(rlim.rlim_max))
        libc.setrlimit(RLIMIT_NOFILE, ctypes.byref(rlim))
    except Exception:
        return


def _debug_post(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    # Disabled in the packaged app: posting to a debug server (127.0.0.1:7777)
    # does a blocking urlopen on every window event, which freezes the GUI while
    # the window is dragged (resized fires dozens of times/sec).
    return
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

    _raise_fd_limit()

    from streamlit.web import bootstrap

    try:
        base_dir = _app_data_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(str(base_dir))
    except Exception:
        pass

    app_py = _resource_path("app.py")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

    flag_options = {
        "global_developmentMode": False,
        "server_headless": True,
        "server_fileWatcherType": "poll",
        "server_runOnSave": False,
        "server_port": port,
        "server_address": "127.0.0.1",
        "browser_gatherUsageStats": False,
        "browser_serverPort": port,
        "browser_serverAddress": "127.0.0.1",
    }

    if not log_path:
        log_path = _prepare_log_file()
    if log_path:
        log_file = Path(log_path)
    else:
        log_file = (Path(tempfile.gettempdir()) / "TFLiteTraining" / "logs" / "streamlit.log").resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch(exist_ok=True)
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
    try { if (typeof win.scheduleLayoutResync === 'function') win.scheduleLayoutResync(); } catch (e) {}
    try { if (typeof win.queueFrameHeightSync === 'function') win.queueFrameHeightSync(); } catch (e) {}
    try { if (typeof win.syncFrameHeight === 'function') win.syncFrameHeight(); } catch (e) {}
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


_LAST_LAYOUT_REFRESH_AT = 0.0


def _schedule_window_layout_refresh(window: "webview.Window", reason: str = "") -> None:
    global _LAST_LAYOUT_REFRESH_AT
    # Debounce: the SPA fires request_reflow many times in quick succession on
    # mount (loaded/shown/mount:t0/t120/t360), and each call would schedule a
    # fresh set of evaluate_js timers that force Streamlit + iframe reflow. Only
    # let one schedule through per window so the native WebView2 is not thrashed.
    now = time.time()
    if (now - _LAST_LAYOUT_REFRESH_AT) < 0.3:
        return
    _LAST_LAYOUT_REFRESH_AT = now
    # #region debug-point C:schedule-window-layout-refresh
    _debug_post("C", "desktop_launcher.py:_schedule_window_layout_refresh", "[DEBUG] shell layout refresh scheduled", {"reason": str(reason or "")})
    # #endregion
    def _run_once(delay_s: float) -> None:
        def _inner() -> None:
            try:
                # #region debug-point C:evaluate-layout-refresh-js
                _debug_post("C", "desktop_launcher.py:_schedule_window_layout_refresh", "[DEBUG] shell evaluate_js layout refresh", {"reason": str(reason or ""), "delay_s": float(delay_s)})
                # #endregion
                window.evaluate_js(_LAYOUT_REFRESH_JS)
            except Exception:
                pass

        timer = threading.Timer(delay_s, _inner)
        timer.daemon = True
        timer.start()

    for delay_s in (0.0, 0.18, 0.5):
        _run_once(delay_s)


_LAST_NATIVE_NUDGE_AT = 0.0


def _maybe_native_resize_nudge(window: "webview.Window", reason: str = "") -> bool:
    global _LAST_NATIVE_NUDGE_AT
    reason_s = str(reason or "")
    if reason_s.startswith("resized:"):
        return False
    if not (reason_s.startswith("image-project-mount") or reason_s in {"shown", "loaded", "startup", "open-project"}):
        return False
    now = time.time()
    if (now - float(_LAST_NATIVE_NUDGE_AT or 0.0)) < 1.2:
        return False
    resize_fn = getattr(window, "resize", None)
    width = int(getattr(window, "width", 0) or 0)
    height = int(getattr(window, "height", 0) or 0)
    if not callable(resize_fn) or width < 300 or height < 300:
        # #region debug-point C:native-resize-nudge-skip
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge skipped", {"reason": reason_s, "width": width, "height": height, "has_resize": bool(callable(resize_fn))})
        # #endregion
        return False
    try:
        _LAST_NATIVE_NUDGE_AT = now
        # #region debug-point C:native-resize-nudge
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge start", {"reason": reason_s, "width": width, "height": height})
        # #endregion
        resize_fn(width + 1, height + 1)
        time.sleep(0.03)
        resize_fn(width, height)
        # #region debug-point C:native-resize-nudge-done
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge done", {"reason": reason_s, "width": width, "height": height})
        # #endregion
        return True
    except Exception as e:
        # #region debug-point C:native-resize-nudge-error
        _debug_post("C", "desktop_launcher.py:_maybe_native_resize_nudge", "[DEBUG] native resize nudge failed", {"reason": reason_s, "error": str(e)})
        # #endregion
        return False


class _ShellApi:
    def __init__(self) -> None:
        self.window = None

    def bind(self, window: "webview.Window") -> None:
        self.window = window

    def request_reflow(self, reason: str = "") -> bool:
        if self.window is None:
            return False
        # #region debug-point C:request-reflow
        _debug_post("C", "desktop_launcher.py:_ShellApi.request_reflow", "[DEBUG] shell request_reflow invoked", {"reason": str(reason or "")})
        # #endregion
        _schedule_window_layout_refresh(self.window, reason=reason)
        _maybe_native_resize_nudge(self.window, reason=reason)
        return True


_SPLASH_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body {
    margin: 0; height: 100%;
    background: #1a212b;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    -webkit-user-select: none; user-select: none;
  }
  body {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px; color: #e8eaed;
  }
  .title { font-size: 18px; font-weight: 800; letter-spacing: -0.01em; }
  .spinner {
    width: 30px; height: 30px; border-radius: 50%;
    border: 3px solid rgba(255,255,255,0.18);
    border-top-color: #8ab4f8;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .sub { font-size: 12px; color: #a8b0ba; }
</style>
</head>
<body>
  <div class="title">TFLiteTraining</div>
  <div class="spinner"></div>
  <div class="sub">Loading…</div>
</body>
</html>"""


def _startup_window_logic(window: "webview.Window") -> None:
    _schedule_window_layout_refresh(window, reason="startup")
    _maybe_native_resize_nudge(window, reason="startup")


def main() -> None:
    multiprocessing.freeze_support()
    # Windows: explicitly use 'spawn' for PyInstaller compatibility
    if os.name == "nt":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set
    _raise_fd_limit()
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = _prepare_log_file()
    proc = multiprocessing.Process(target=_run_streamlit_server, args=(port, log_path), daemon=True)
    proc.start()
    _startup_log(f"streamlit child started (pid={proc.pid}, port={port})")
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
        _startup_log("streamlit ready (urlopen 200)")

        import webview

        shell_api = _ShellApi()
        _startup_log("creating window")
        window = webview.create_window("TF Lite Training", url, width=1200, height=800, js_api=shell_api)
        shell_api.bind(window)
        # NOTE: these handlers must stay cheap — `resized` fires dozens of times
        # per second while the window is dragged, and any blocking work (network
        # debug posts, file logging) here freezes the GUI. `_schedule_window_layout_refresh`
        # is debounced to 0.3s, so it is safe.
        window.events.loaded += lambda: _schedule_window_layout_refresh(window, reason="loaded")
        window.events.shown += lambda: _schedule_window_layout_refresh(window, reason="shown")
        window.events.restored += lambda: _schedule_window_layout_refresh(window, reason="restored")
        window.events.maximized += lambda: _schedule_window_layout_refresh(window, reason="maximized")
        window.events.resized += lambda width, height: _schedule_window_layout_refresh(window, reason=f"resized:{width}x{height}")
        window.events.closed += lambda: _shutdown_and_exit(proc)
        _startup_log("entering webview.start (GUI loop)")
        webview.start(_startup_window_logic, window, private_mode=False)
        _startup_log("webview.start returned (window closed)")
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
