from __future__ import annotations

import argparse
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
    """Return a writable log path; fall back to /tmp if the app-data dir fails."""
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
    # Absolute last resort — stderr
    return ""


def _configure_multiprocessing_executable() -> None:
    if sys.platform != "darwin":
        return
    if not getattr(sys, "frozen", False):
        return
    helper = Path(sys.executable).with_name("TFLiteTrainingConsole")
    if helper.exists():
        multiprocessing.set_executable(str(helper))


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
    import signal as _signal

    def _on_terminate_signal(signum, frame):
        # Window closed → the launcher calls proc.terminate() → SIGTERM.
        # The default handler would kill this process WITHOUT running
        # atexit, so the camera bridge would never release its
        # AVCaptureSessions (camera stays claimed, pthread trap on
        # finalization).  Release them here while the interpreter is fully
        # alive, then exit normally so every other atexit handler
        # (controller / server / serial cleanup) runs too.
        try:
            import mac_camera

            mac_camera.shutdown_all_caps(wait_s=0.2)
        except Exception:
            pass
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _on_terminate_signal)
    _signal.signal(_signal.SIGINT, _on_terminate_signal)

    with log_file.open("a", encoding="utf-8") as f:
        try:
            import resource

            f.write(f"[TFLiteTraining] RLIMIT_NOFILE={resource.getrlimit(resource.RLIMIT_NOFILE)}\n")
            f.flush()
        except Exception:
            pass
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                bootstrap.load_config_options(flag_options=flag_options)
                try:
                    from streamlit import config as st_config

                    f.write(f"[TFLiteTraining] server.fileWatcherType={st_config.get_option('server.fileWatcherType')}\n")
                    f.flush()
                except Exception:
                    pass
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


def _schedule_window_layout_refresh(window: "webview.Window", reason: str = "") -> None:
    def _run_once(delay_s: float) -> None:
        def _inner() -> None:
            try:
                _dispatch_ui(lambda: window.evaluate_js(_LAYOUT_REFRESH_JS))
            except Exception:
                pass

        timer = threading.Timer(delay_s, _inner)
        timer.daemon = True
        timer.start()

    for delay_s in (0.0, 0.12, 0.35, 0.8):
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
        return False
    def _nudge() -> None:
        try:
            resize_fn(width + 1, height + 1)
            time.sleep(0.03)
            resize_fn(width, height)
        except Exception:
            pass

    _LAST_NATIVE_NUDGE_AT = now
    return _dispatch_ui(_nudge)


def _force_exit(code: int = 0) -> None:
    """Terminate the app without Python shutdown hooks hanging on webview."""
    os._exit(int(code))


def _dispatch_ui(callback) -> bool:
    """Run a window mutation on the Cocoa main thread, never synchronously.

    A pywebview.api method runs on the JS-bridge thread.  Calling AppKit /
    NSWindow directly from there while WebKit waits for the bridge can
    deadlock the UI (beachball).  PyObjCTools.AppHelper.callAfter is the
    non-blocking main-queue hop; dispatch_sync is deliberately never used.
    """
    if sys.platform == "darwin":
        # Already on the Cocoa main thread (e.g. a window event or a test)?
        # Call directly; AppHelper.callAfter would queue a callback that only
        # runs when the main loop is idle.
        try:
            import threading

            if threading.current_thread() is threading.main_thread():
                callback()
                return True
        except Exception:
            pass
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(callback)
            return True
        except Exception:
            pass
    try:
        callback()
        return True
    except Exception:
        return False


class _ShellApi:
    def __init__(self, controller=None) -> None:
        self.window = None
        self.controller = controller
        self._compact_size = (1040, 680)
        self._window_mode = "launcher"
        self._drag_origin = None
        self._menu_target = None
        self._resize_frame = None
        self._resize_edge = "e"

    def bind(self, window: "webview.Window") -> None:
        self.window = window

    def quit_app(self) -> bool:
        """Expose `window.pywebview.api.quit_app()` to the custom × button.

        Destroys the frameless window, releases camera/serial/server threads,
        and hard-exits the process so no background controller survives.
        """
        window = self.window
        if window is not None:
            # destroy() is an NSWindow operation; run it on the Cocoa main
            # thread, never on the JS-bridge thread that called this method.
            _dispatch_ui(lambda: window.destroy())

        if self.controller is not None:
            try:
                self.controller.release_all_resources()
            except Exception:
                pass
        # The window event may already exit; this is the guaranteed path for
        # a WKWebView callback thread that never returns to the main loop.
        threading.Timer(0.1, _force_exit).start()
        return True

    def _visible_screen_rect(self):
        """(x, y, width, height) of the active display's usable area."""
        if sys.platform == "darwin":
            try:
                from AppKit import NSScreen

                frame = NSScreen.mainScreen().visibleFrame()
                return (
                    int(frame.origin.x),
                    int(frame.origin.y),
                    int(frame.size.width),
                    int(frame.size.height),
                )
            except Exception:
                pass
        try:
            import webview as _webview

            screen = _webview.screens[0]
            return (
                int(getattr(screen, "x", 0) or 0),
                int(getattr(screen, "y", 0) or 0),
                int(getattr(screen, "width", 1440) or 1440),
                int(getattr(screen, "height", 900) or 900),
            )
        except Exception:
            return None

    def _full_screen_rect(self):
        """(x, y, width, height) of the whole display, menu bar/Dock included."""
        if sys.platform == "darwin":
            try:
                from AppKit import NSScreen

                frame = NSScreen.mainScreen().frame()
                return (
                    int(frame.origin.x),
                    int(frame.origin.y),
                    int(frame.size.width),
                    int(frame.size.height),
                )
            except Exception:
                pass
        try:
            import webview as _webview

            screen = _webview.screens[0]
            return (
                int(getattr(screen, "x", 0) or 0),
                int(getattr(screen, "y", 0) or 0),
                int(getattr(screen, "width", 1440) or 1440),
                int(getattr(screen, "height", 900) or 900),
            )
        except Exception:
            return self._visible_screen_rect()

    @staticmethod
    def _set_presentation(fullscreen: bool = False) -> None:
        """Keep AppKit at the standard presentation (NO kiosk flags)."""
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSApplication, NSApplicationPresentationDefault

            options = NSApplicationPresentationDefault
            NSApplication.sharedApplication().setPresentationOptions_(options)
            _direct_log(f"presentation options={int(options)}")
        except Exception as exc:
            _direct_log(f"presentation options failed: {exc}")

    @staticmethod
    def _show_system_menu_bar(label: str = "") -> None:
        """Force the macOS menu bar visible (native fullscreen auto-hides it)."""
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSMenu

            NSMenu.setMenuBarVisible_(True)
            if label:
                _direct_log(f"menu bar visible · {label}")
        except Exception as exc:
            _direct_log(f"menu bar visible failed: {exc}")

    @staticmethod
    def _apply_launcher_style(native) -> None:
        """Pure Borderless (mask 0) + opaque square content for Launcher."""
        if native is None:
            return
        try:
            from AppKit import NSWindowStyleMaskBorderless

            native.setStyleMask_(NSWindowStyleMaskBorderless)
        except Exception:
            pass
        try:
            native.setOpaque_(True)
        except Exception:
            pass
        try:
            native.setHasShadow_(True)
        except Exception:
            pass
        try:
            view = native.contentView()
            if view is not None:
                view.setWantsLayer_(True)
                layer = view.layer()
                if layer is not None:
                    layer.setCornerRadius_(0.0)
                    layer.setMasksToBounds_(True)
        except Exception:
            pass
        try:
            layer = native.layer()
            if layer is not None:
                layer.setCornerRadius_(0.0)
        except Exception:
            pass

    @staticmethod
    def _apply_workspace_style(native) -> None:
        """Titled resizable full-size window; traffic lights visible/enabled."""
        if native is None:
            return
        try:
            from AppKit import (
                NSWindowCollectionBehaviorFullScreenPrimary,
                NSWindowStyleMaskClosable,
                NSWindowStyleMaskFullSizeContentView,
                NSWindowStyleMaskMiniaturizable,
                NSWindowStyleMaskResizable,
                NSWindowStyleMaskTitled,
                NSWindowTitleHidden,
            )

            mask = (
                NSWindowStyleMaskTitled
                | NSWindowStyleMaskClosable
                | NSWindowStyleMaskMiniaturizable
                | NSWindowStyleMaskResizable
                | NSWindowStyleMaskFullSizeContentView
            )
            native.setStyleMask_(mask)
            try:
                native.setTitleVisibility_(NSWindowTitleHidden)
                native.setTitlebarAppearsTransparent_(True)
            except Exception:
                pass
            native.setCollectionBehavior_(NSWindowCollectionBehaviorFullScreenPrimary)
            native.setOpaque_(True)
            for button_index in (0, 1, 2):  # close, miniaturize, zoom
                try:
                    button = native.standardWindowButton_(button_index)
                    if button is not None:
                        button.setHidden_(False)
                        button.setEnabled_(True)
                except Exception:
                    pass
        except Exception:
            pass

    def _log_window_state(self, label: str) -> None:
        window = self.window
        native = getattr(window, "native", None) if window is not None else None
        if native is None and window is not None:
            native = getattr(window, "_window", None)
        if native is None:
            _direct_log(f"{label}: native window unavailable")
            return

        def _log() -> None:
            try:
                from AppKit import NSApplication, NSWindowStyleMaskFullScreen

                mask = int(native.styleMask())
                fullscreen = bool(mask & NSWindowStyleMaskFullScreen)
                try:
                    presentation = int(NSApplication.sharedApplication().presentationOptions())
                except Exception:
                    presentation = 0
                try:
                    close_button = native.standardWindowButton_(0)
                    close_enabled = bool(
                        close_button is not None
                        and not close_button.isHidden()
                        and close_button.isEnabled()
                    )
                except Exception:
                    close_enabled = False
                if label == "Launcher":
                    _direct_log(
                        f"Launcher: styleMask={mask}, fullscreen={fullscreen} "
                        f"(Borderless), presentationOptions={presentation}"
                    )
                else:
                    _direct_log(
                        "Workspace: "
                        f"NSWindowStyleMaskFullScreen={fullscreen}, "
                        f"presentationOptions={presentation}, "
                        f"closeButtonEnabled={close_enabled}, styleMask={mask}"
                    )
            except Exception as exc:
                _direct_log(f"{label} state log failed: {exc}")

        _dispatch_ui(_log)

    def _configure_native_window(self, window: "webview.Window") -> bool:
        """Force the Launcher's non-fullscreen window to pure Borderless."""
        if sys.platform != "darwin":
            return False
        native = getattr(window, "native", None) or getattr(window, "_window", None)
        if native is None:
            return False

        def _apply() -> None:
            self._apply_launcher_style(native)
            self._log_window_state("Launcher")

        _dispatch_ui(_apply)
        return True

    @staticmethod
    def _apply_native_rect(native, x: int, y: int, width: int, height: int) -> bool:
        if native is None or not hasattr(native, "setFrame_display_"):
            return False
        try:
            from AppKit import NSMakeRect

            native.setFrame_display_(NSMakeRect(int(x), int(y), int(width), int(height)), True)
            return True
        except Exception:
            return False

    def _pywebview_top_y(self, y_bottom: int, height: int) -> int:
        """Convert a Cocoa bottom-left y into pywebview's top-origin y.

        pywebview's cocoa backend implements ``move(x, y)`` as
        ``flipped_y = screen.height - y`` against the FULL screen frame.
        Callers here hold ``visibleFrame`` coordinates (bottom-left origin,
        menu bar / Dock excluded), so passing ``y`` straight through shifted
        the window down by the Dock offset — the off-screen-to-the-bottom
        bug.  Resize first (it preserves the top-left), then move with this
        converted value.
        """
        try:
            import webview as _webview

            full_height = int(getattr(_webview.screens[0], "height", 0) or 0)
            if full_height > 0:
                return max(0, full_height - (int(y_bottom) + int(height)))
        except Exception:
            pass
        return int(y_bottom)

    @staticmethod
    def _set_native_frame(native, x: int, y: int, width: int, height: int) -> bool:
        """Set a Cocoa frame in its native bottom-left coordinate space."""
        if native is None or not hasattr(native, "setFrame_display_"):
            return False
        try:
            from AppKit import NSMakeRect

            native.setFrame_display_(NSMakeRect(int(x), int(y), int(width), int(height)), True)
            try:
                native.makeKeyAndOrderFront_(None)
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _center_launcher_frame(self, native, window, width: int = 1040, height: int = 680) -> bool:
        rect = self._visible_screen_rect()
        if rect is not None:
            sx, sy, sw, sh = rect
            x = sx + max(0, (sw - width) // 2)
            y = sy + max(0, (sh - height) // 2)
            if self._set_native_frame(native, x, y, width, height):
                return True
            try:
                window.resize(width, height)
                window.move(x, self._pywebview_top_y(y, height))
                return True
            except Exception:
                pass
        try:
            window.resize(width, height)
        except Exception:
            pass
        return False

    def _finish_launcher_restore(self) -> None:
        """Strip to Borderless and restore the centered Launcher frame."""
        window = self.window
        if window is None:
            return
        native = getattr(window, "native", None) or getattr(window, "_window", None)
        self._window_mode = "launcher"
        self._resize_frame = None

        def _apply() -> None:
            try:
                from AppKit import NSWindowStyleMaskFullScreen

                if native is not None and (
                    int(native.styleMask()) & NSWindowStyleMaskFullScreen
                ):
                    # The exit animation has not finished yet; wait for the
                    # notification, with a bounded retry as a safety net.
                    native.toggleFullScreen_(None)
                    timer = threading.Timer(0.8, self._finish_launcher_restore)
                    timer.daemon = True
                    timer.start()
                    return
            except Exception:
                pass
            self._apply_launcher_style(native)
            self._center_launcher_frame(native, window, 1040, 680)
            self._log_window_state("Launcher")

        _dispatch_ui(_apply)

    def on_window_entered_fullscreen(self) -> None:
        """windowDidEnterFullScreen_: ensure the native traffic lights remain."""
        if self._window_mode != "workspace" or self.window is None:
            return
        native = getattr(self.window, "native", None) or getattr(self.window, "_window", None)
        if native is None:
            return

        def _apply() -> None:
            for button_index in (0, 1, 2):
                try:
                    button = native.standardWindowButton_(button_index)
                    if button is not None:
                        button.setHidden_(False)
                        button.setEnabled_(True)
                except Exception:
                    pass
            self._show_system_menu_bar("workspace fullscreen")
            self._log_window_state("Workspace")

        _dispatch_ui(_apply)

    def on_window_restored(self, _retry: int = 0) -> None:
        """windowDidExitFullScreen_.

        Inside a project the green button only leaves *native fullscreen*; it
        must NOT strip the titlebar to Borderless (that removes the red /
        yellow / green buttons and there is then no way back).  Only an
        explicit enter_launcher_window() returns to the sharp Launcher frame.
        """
        window = self.window
        if window is None:
            return
        native = getattr(window, "native", None) or getattr(window, "_window", None)

        def _apply() -> None:
            try:
                from AppKit import NSWindowStyleMaskFullScreen

                if native is not None and (
                    int(native.styleMask()) & NSWindowStyleMaskFullScreen
                ):
                    # The notification can arrive while the exit animation
                    # still reports the FullScreen bit.  Re-check instead of
                    # bailing out — bailing left a workspace window with no
                    # traffic lights and no way back into fullscreen.
                    if _retry < 8:
                        timer = threading.Timer(
                            0.15, lambda: self.on_window_restored(_retry + 1)
                        )
                        timer.daemon = True
                        timer.start()
                    return
            except Exception:
                pass
            if self._window_mode == "workspace":
                self._apply_workspace_style(native)
                for button_index in (0, 1, 2):
                    try:
                        button = native.standardWindowButton_(button_index) if native is not None else None
                        if button is not None:
                            button.setHidden_(False)
                            button.setEnabled_(True)
                    except Exception:
                        pass
                self._show_system_menu_bar("workspace windowed")
                self._log_window_state("Workspace")
                return
            self._finish_launcher_restore()

        _dispatch_ui(_apply)

    def enter_workspace_window(self) -> bool:
        """Enter real macOS native fullscreen with system traffic lights."""
        window = self.window
        if window is None:
            return False
        self._compact_size = (1040, 680)
        self._window_mode = "workspace"
        self._drag_origin = None
        self._resize_frame = None

        def _apply() -> None:
            self._set_presentation(False)
            native = getattr(window, "native", None) or getattr(window, "_window", None)
            if sys.platform == "darwin" and native is not None:
                self._apply_workspace_style(native)
                try:
                    from AppKit import NSWindowStyleMaskFullScreen

                    if not (int(native.styleMask()) & NSWindowStyleMaskFullScreen):
                        native.toggleFullScreen_(None)
                    else:
                        native.makeKeyAndOrderFront_(None)
                except Exception:
                    pass
                # windowDidEnterFullScreen_ fires events.maximized, which
                # logs the settled FullScreen=True state (proof line).
                self._log_window_state("Workspace")
                return
            # Non-Cocoa fallback: fill the screen frame.
            rect = self._full_screen_rect()
            if rect is not None:
                x, y, width, height = rect
                try:
                    window.resize(width, height)
                    window.move(x, self._pywebview_top_y(y, height))
                    return
                except Exception:
                    pass
            try:
                window.maximize()
            except Exception:
                pass

        _dispatch_ui(_apply)
        return True

    def enter_launcher_window(self) -> bool:
        """Restore the centered 1040×680 sharp-cornered Launcher frame."""
        window = self.window
        if window is None:
            return False
        self._window_mode = "launcher"
        self._drag_origin = None
        self._resize_frame = None
        width, height = 1040, 680

        def _apply() -> None:
            self._set_presentation(False)
            try:
                window.restore()
            except Exception:
                pass
            native = getattr(window, "native", None) or getattr(window, "_window", None)
            if sys.platform == "darwin" and native is not None:
                try:
                    from AppKit import NSWindowStyleMaskFullScreen

                    if int(native.styleMask()) & NSWindowStyleMaskFullScreen:
                        # Let windowDidExitFullScreen_ strip/recenter after
                        # the native exit animation.
                        native.toggleFullScreen_(None)
                        return
                except Exception:
                    pass
                self._apply_launcher_style(native)
                self._center_launcher_frame(native, window, width, height)
                self._log_window_state("Launcher")
                return
            self._center_launcher_frame(None, window, width, height)

        _dispatch_ui(_apply)
        return True

    def toggle_fullscreen(self) -> bool:
        """Ctrl+Cmd+F / View > Toggle Full Screen."""
        window = self.window
        if window is None:
            return False
        if self._window_mode == "launcher":
            return self.enter_workspace_window()
        self._window_mode = "workspace"

        def _apply() -> None:
            native = getattr(window, "native", None) or getattr(window, "_window", None)
            if sys.platform == "darwin" and native is not None:
                try:
                    self._apply_workspace_style(native)
                    native.toggleFullScreen_(None)
                    return
                except Exception:
                    pass
            try:
                window.toggle_fullscreen()
            except Exception:
                pass

        _dispatch_ui(_apply)
        return True

    # Back-compat names used by older shell builds.
    def enter_workspace(self) -> bool:
        return self.enter_workspace_window()

    def exit_workspace(self) -> bool:
        return self.enter_launcher_window()

    def begin_window_drag(self) -> bool:
        """Capture the frame origin for a Launcher-only custom window drag."""
        if self._window_mode != "launcher" or self.window is None:
            return False
        window = self.window
        self._drag_origin = None

        def _capture() -> None:
            try:
                x = getattr(window, "x", None)
                y = getattr(window, "y", None)
                if x is not None and y is not None:
                    self._drag_origin = (int(x), int(y))
                    return
            except Exception:
                pass
            try:
                native = getattr(window, "native", None) or getattr(window, "_window", None)
                frame = native.frame() if native is not None else None
                if frame is not None:
                    self._drag_origin = (int(frame.origin.x), int(frame.origin.y))
            except Exception:
                pass

        _dispatch_ui(_capture)
        return True

    def window_drag_to(self, dx=0, dy=0) -> bool:
        """Move the window by a cursor delta from `begin_window_drag`."""
        if self._window_mode != "launcher" or self.window is None:
            return False
        window = self.window
        offset_x = int(float(dx or 0))
        offset_y = int(float(dy or 0))

        def _apply() -> None:
            origin = self._drag_origin
            if origin is None:
                return
            target_x = int(origin[0]) + offset_x
            target_y = int(origin[1]) + offset_y
            try:
                native = getattr(window, "native", None) or getattr(window, "_window", None)
                if sys.platform == "darwin" and native is not None and hasattr(native, "setFrameOrigin_"):
                    native.setFrameOrigin_((target_x, target_y))
                    return
            except Exception:
                pass
            try:
                window.move(target_x, target_y)
            except Exception:
                pass

        _dispatch_ui(_apply)
        return True

    def begin_window_resize(self, edge: str = "e") -> bool:
        """Capture the Launcher's frame for an HTML resize-grip drag."""
        if self._window_mode != "launcher" or self.window is None:
            return False
        if sys.platform != "darwin":
            return False
        window = self.window
        native = getattr(window, "native", None) or getattr(window, "_window", None)
        if native is None or not hasattr(native, "frame"):
            return False
        self._resize_edge = str(edge or "e").lower()
        self._resize_frame = None

        def _capture() -> None:
            try:
                frame = native.frame()
                self._resize_frame = (
                    int(frame.origin.x),
                    int(frame.origin.y),
                    int(frame.size.width),
                    int(frame.size.height),
                )
            except Exception:
                self._resize_frame = None

        _dispatch_ui(_capture)
        return True

    def window_resize_to(self, dx=0, dy=0) -> bool:
        """Resize the Launcher from cumulative pointer deltas (top-origin).

        HTML grips exist because pywebview's borderless NSWindow ignores
        ``resizable=True`` for edge drags.  Deltas are measured from the
        pointer-down point; min_size is enforced here, not by pywebview.
        """
        if self._window_mode != "launcher" or self.window is None:
            return False
        if self._resize_frame is None:
            return False
        try:
            offset_x = int(float(dx or 0))
            offset_y = int(float(dy or 0))
        except Exception:
            return False
        edge = self._resize_edge
        window = self.window

        def _apply() -> None:
            frame = self._resize_frame
            if frame is None:
                return
            x, y, width, height = frame
            min_width, min_height = 960, 640
            if "e" in edge:
                width = max(min_width, width + offset_x)
            if "w" in edge:
                new_width = max(min_width, width - offset_x)
                x += width - new_width
                width = new_width
            if "n" in edge:
                # Top edge: drag up (negative dy) grows the window upward.
                height = max(min_height, height - offset_y)
            if "s" in edge:
                # Bottom edge: drag down (positive dy) lowers the bottom.
                new_height = max(min_height, height + offset_y)
                y += height - new_height
                height = new_height
            native = getattr(window, "native", None) or getattr(window, "_window", None)
            self._apply_native_rect(native, x, y, width, height)

        _dispatch_ui(_apply)
        return True

    def request_reflow(self, reason: str = "") -> bool:
        if self.window is None:
            return False
        _schedule_window_layout_refresh(self.window, reason=reason)
        _maybe_native_resize_nudge(self.window, reason=reason)
        return True

    def pick_upload_files(self, base_url: str, session: str, class_name: str) -> list:
        """Native file picker for the SPA upload button.

        The SPA's <input type="file"> does not reliably open a dialog inside
        pywebview on macOS, so the SPA calls this JS API instead: open the
        native open-file dialog, read the chosen images, and POST them to
        the local controller's /upload endpoint.

        Returns a list of per-file results:
          {"filename": str, "saved_filename": str, "ok": bool, "error": str, "thumb_b64": str}
        so the SPA can show a detailed summary instead of a single success toast.
        """
        import base64 as _b64
        import pathlib as _pl
        import urllib.error as _ue

        results = []
        try:
            import webview as _webview

            paths = _webview.create_file_dialog(
                _webview.OPEN_DIALOG,
                allow_multiple=True,
                file_types=("Image files (*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp)",),
            )
            if not paths:
                return results
            base_url = str(base_url or "").strip()
            session = str(session or "").strip()
            class_name = str(class_name or "").strip()
            if not base_url or not session or not class_name:
                results.append({"filename": "", "saved_filename": "", "ok": False, "error": "missing upload context (reload page)", "thumb_b64": ""})
                return results
            endpoint = f"{base_url}/upload"
            for p in paths:
                fpath = _pl.Path(str(p))
                fname = fpath.name
                saved = ""
                try:
                    data = fpath.read_bytes()
                except Exception as e:
                    results.append({"filename": fname, "saved_filename": "", "ok": False, "error": f"read failed: {e}", "thumb_b64": ""})
                    continue
                payload = {
                    "session": session,
                    "class": class_name,
                    "image_b64": _b64.b64encode(data).decode("ascii"),
                    "filename": fname,
                }
                ok = False
                err = ""
                thumb = ""
                try:
                    req = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        raw = resp.read()
                        body = json.loads(raw.decode("utf-8") or "{}")
                    ok = str(body.get("ok") or "0") == "1"
                    err = "" if ok else str(body.get("error") or "upload rejected")
                    thumb = str(body.get("thumb_b64") or "")
                    saved = str(body.get("filename") or "")
                except _ue.HTTPError as he:
                    err_body = ""
                    try:
                        raw = he.read()
                        err_body = json.loads(raw.decode("utf-8") or "{}").get("error", "")
                    except Exception:
                        err_body = ""
                    ok = False
                    err = f"server HTTP {he.code}: {err_body or he.reason}"
                    thumb = ""
                except _ue.URLError as ue:
                    ok = False
                    err = f"network unreachable: {ue.reason}"
                    thumb = ""
                except Exception as e:
                    ok = False
                    err = f"upload failed: {e}"
                    thumb = ""
                results.append({
                    "filename": fname,
                    "saved_filename": saved,
                    "ok": bool(ok),
                    "error": str(err or ""),
                    "thumb_b64": thumb if ok else "",
                })
        except Exception as e:
            results.append({"filename": "", "saved_filename": "", "ok": False, "error": f"picker failed: {e}", "thumb_b64": ""})
        return results

    def pick_single_image_file(self) -> dict:
        """Native single-image picker for the preview Upload-File source."""
        import base64 as _b64
        import pathlib as _pl
        import mimetypes as _mt

        try:
            import webview as _webview

            paths = _webview.create_file_dialog(
                _webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("Image files (*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp)",),
            )
            if not paths:
                return {}
            fpath = _pl.Path(str(paths[0]))
            data = fpath.read_bytes()
            mime = _mt.guess_type(fpath.name)[0] or "image/png"
            return {"name": fpath.name, "mime": mime, "b64": _b64.b64encode(data).decode("ascii")}
        except Exception:
            return {}


def _startup_window_logic(window: "webview.Window") -> None:
    _schedule_window_layout_refresh(window, reason="startup")
    _maybe_native_resize_nudge(window, reason="startup")


def _run_legacy_streamlit() -> None:
    """Rollback path: the pre-Phase-1 Streamlit host, kept behind --legacy-ui."""
    _configure_multiprocessing_executable()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass
    _raise_fd_limit()
    multiprocessing.freeze_support()
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = _prepare_log_file()
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

        shell_api = _ShellApi()
        window = webview.create_window("TF Lite Training", url, width=1200, height=800, js_api=shell_api)
        shell_api.bind(window)
        window.events.loaded += lambda: _schedule_window_layout_refresh(window, reason="loaded")
        window.events.shown += lambda: _schedule_window_layout_refresh(window, reason="shown")
        window.events.restored += lambda: _schedule_window_layout_refresh(window, reason="restored")
        window.events.maximized += lambda: _schedule_window_layout_refresh(window, reason="maximized")
        window.events.resized += lambda width, height: _schedule_window_layout_refresh(window, reason=f"resized:{width}x{height}")
        window.events.closed += lambda: _shutdown_and_exit(proc)
        webview.start(_startup_window_logic, window)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def _direct_log_path() -> Path:
    try:
        path = _app_data_dir() / "logs" / "aioscout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        path = Path(tempfile.gettempdir()) / "TFLiteTraining" / "logs" / "aioscout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _direct_log(message: str) -> None:
    try:
        with _direct_log_path().open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


_MENU_SPEC = (
    ("File", (
        ("new", "New Project…"),
        ("open", "Open .tmproj…"),
        ("open-backup", "Open Backup File…"),
        None,
        ("save", "Save"),
        ("save-as", "Save As…"),
        None,
        ("export-dataset", "Export Dataset…"),
        ("export", "Export Model…"),
        None,
        ("quit", "Quit AIoScout"),
    )),
    ("Edit", (
        ("inspect-class", "Inspect Active Class"),
        ("rename", "Rename Active Class"),
        ("delete-class", "Delete Active Class"),
        None,
        ("edit-chars", "Edit Characters…"),
        ("emnist-settings", "EMNIST Settings…"),
        ("plates", "Manage Plate Whitelist…"),
        None,
        ("reset", "Reset Project"),
    )),
    ("View", (
        ("view-collect", "01 Collect"),
        ("view-edit", "02 Edit"),
        ("view-train", "03 Train"),
        ("view-preview", "04 Preview"),
        None,
        ("toggle-fullscreen", "Toggle Full Screen  (⌃⌘F)"),
        ("toggle-theme", "Toggle Light/Dark Mode"),
    )),
    ("Settings", (
        ("project-settings", "Project Settings…"),
        ("global-settings", "Global Settings…"),
        None,
        ("close-project", "Close Project"),
    )),
    ("Help", (
        ("tutorial", "Tutorial"),
        ("eula", "EULA"),
        ("licenses", "Third-Party Licenses"),
    )),
)


def _build_macos_menu(window: "webview.Window") -> list:
    """Mirror window.AIoScoutNativeMenu's ids into a real NSMenu.

    pywebview's MenuAction has no enabled/shortcut fields; the shell owns
    enabled state and keyboard accelerators.  Every action dispatches the
    same command id as the in-app menu and ⌘K palette.
    """
    try:
        from webview.menu import Menu, MenuAction, MenuSeparator
    except Exception:
        return []

    def dispatch(command_id: str) -> None:
        try:
            payload = json.dumps(str(command_id))
            window.evaluate_js(f"window.AIoScoutNativeMenu && window.AIoScoutNativeMenu.dispatch({payload})")
        except Exception:
            pass

    # `__app__` items are inserted between About and Services by pywebview.
    # The app menu title itself is renamed to AIoScout after startup.
    menus = [Menu("__app__", [MenuAction("Settings…", lambda: dispatch("global-settings"))])]
    for title, items in _MENU_SPEC:
        entries = []
        for item in items:
            if item is None:
                entries.append(MenuSeparator())
                continue
            command_id, label = item
            entries.append(MenuAction(label, lambda cid=command_id: dispatch(cid)))
        menus.append(Menu(title, entries))
    return menus


def _run_direct_ui() -> None:
    """Serve the shell + API from one local origin in a frameless window.

    The only js_api surface is the quit hook; all data/layout flows over HTTP
    and the in-app #fsBrowserModal (no native dialogs for picking folders).
    """
    _raise_fd_limit()
    from record_controller import RecordController

    controller = RecordController()
    controller.start()
    port = int(controller.port)
    shell_path = getattr(controller, "_shell_path", None)
    _direct_log(f"direct launch · http://127.0.0.1:{port}/ · shell={shell_path}")
    if shell_path is None:
        try:
            controller.release_all_resources()
        except Exception:
            pass
        raise RuntimeError(
            "AIoScout_UI_Shell.html was not found. Set AIOSCOUT_UI_SHELL to the shell path "
            "or run from the repository root."
        )
    import webview

    # pywebview otherwise inserts its own View + Edit menus BEFORE ours,
    # producing Python | Edit | View | File | Edit | View | Settings | Help.
    try:
        webview.settings["SHOW_DEFAULT_MENUS"] = False
    except Exception:
        pass
    try:
        from AppKit import NSProcessInfo

        NSProcessInfo.processInfo().setProcessName_("AIoScout")
    except Exception:
        pass

    shell_api = _ShellApi(controller)
    menu = []
    window = webview.create_window(
        "AIoScout — TFLite Training",
        f"http://127.0.0.1:{port}/",
        width=1040,
        height=680,
        min_size=(960, 640),
        resizable=True,
        frameless=True,
        easy_drag=False,
        js_api=shell_api,
    )
    shell_api.bind(window)
    controller.set_quit_handler(shell_api.quit_app)
    controller.set_window_mode_handler(
        lambda mode: shell_api.enter_workspace_window()
        if str(mode) == "workspace"
        else shell_api.enter_launcher_window()
    )
    if sys.platform == "darwin":
        menu = _build_macos_menu(window)
        shell_api._menu_target = menu
        try:
            window.menu = menu
        except Exception:
            pass

    def _on_loaded() -> None:
        # Launcher is pure Borderless (mask 0) with an opaque, square content
        # view; native fullscreen is entered only by enter_workspace_window.
        shell_api._configure_native_window(window)
        if shell_api._window_mode == "launcher":
            shell_api.enter_launcher_window()
        if sys.platform == "darwin":
            try:
                from AppKit import NSApplication

                main_menu = NSApplication.sharedApplication().mainMenu()
                if main_menu is not None and main_menu.numberOfItems() > 0:
                    first = main_menu.itemAtIndex_(0)
                    first.setTitle_("AIoScout")
                    submenu = first.submenu()
                    if submenu is not None:
                        submenu.setTitle_("AIoScout")
                    titles = [
                        str(main_menu.itemAtIndex_(index).title())
                        for index in range(main_menu.numberOfItems())
                    ]
                    _direct_log("menu: " + " | ".join(titles))
            except Exception as exc:
                _direct_log(f"menu inspect failed: {exc}")
            shell_api._show_system_menu_bar("loaded")
        if sys.platform == "darwin":
            try:
                from AppKit import NSWindowDidExitFullScreenNotification
                from Foundation import NSNotificationCenter

                native = getattr(window, "native", None) or getattr(window, "_window", None)
                if native is not None:
                    NSNotificationCenter.defaultCenter().addObserverForName_object_queue_usingBlock_(
                        NSWindowDidExitFullScreenNotification,
                        native,
                        None,
                        lambda note: shell_api.on_window_restored(),
                    )
            except Exception as exc:
                _direct_log(f"fullscreen exit observer failed: {exc}")
        mounted = "true" if sys.platform == "darwin" else "false"
        try:
            window.evaluate_js(f"window.AIoScoutNativeMenu && window.AIoScoutNativeMenu.setNativeMenuMounted({mounted})")
        except Exception:
            pass

    def _on_closing() -> None:
        # The system 🔴 Close button goes through pywebview's
        # windowShouldClose_ -> this event.  Release camera/serial BEFORE the
        # window disappears so the LED goes out and no worker survives.
        try:
            controller.release_all_resources()
        except Exception:
            pass
        _direct_log("direct launch · closing via window control")

    def _on_closed() -> None:
        try:
            controller.release_all_resources()
        except Exception:
            pass
        _direct_log("direct launch · closed")
        _force_exit(0)

    window.events.loaded += _on_loaded
    window.events.closing += _on_closing
    window.events.closed += _on_closed
    window.events.maximized += lambda: shell_api.on_window_entered_fullscreen()
    window.events.restored += lambda: shell_api.on_window_restored()
    try:
        webview.start(menu=menu)
    finally:
        try:
            controller.release_all_resources()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--legacy-ui", action="store_true", help="Use the Streamlit host (rollback path)")
    args, _unknown = parser.parse_known_args()
    if args.legacy_ui:
        _run_legacy_streamlit()
        return
    _run_direct_ui()


def _shutdown_and_exit(proc: multiprocessing.Process) -> None:
    if proc.is_alive():
        # SIGTERM → the child's handler releases the camera sessions and
        # exits gracefully (its atexit handlers run).
        proc.terminate()
        proc.join(timeout=5)
    if proc.is_alive():
        # The child ignored/hung on SIGTERM (e.g. blocked in a camera read).
        # SIGKILL it so it can never outlive the window and keep the camera
        # claimed — the OS reclaims the capture hardware on process death.
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(timeout=3)
    os._exit(0)



if __name__ == "__main__":
    main()
