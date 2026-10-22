"""Windows-compatibility smoke test (runs on ANY host).

Windows builds of this app live on the `windows-compat` branch.  The Mac
line (Mac-New) and the Windows line share one codebase, but Windows carries
adaptations the Mac line must never silently regress: native ctypes file
dialogs, CAP_DSHOW/CAP_MSMF camera probing, COM-port fallback, WebView2
runtime quirks, platform-marked requirements and explicit UTF-8 I/O.

This suite simulates the win32 checks STATICALLY (no Windows machine
needed) and asserts every one of those adaptations is still present after
a merge from Mac-New.  It is the tripwire that makes a lost Windows fix a
test failure instead of a field report.

Run:
    .venv/bin/python tests/test_windows_compat_smoke.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PASSED = 0
_FAILED = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSED
    if cond:
        _PASSED += 1
        print(f"ok   {name}")
    else:
        _FAILED.append((name, detail))
        print(f"FAIL {name}  {detail}")


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _has(path: str, needle: str) -> bool:
    return needle in _text(path)


# ── 1. requirements: no darwin-only packages leak onto Windows ────────────
def _check_requirements() -> None:
    req = _text("requirements.txt")
    _check("req: imageio-ffmpeg kept (Windows camera names)", "imageio-ffmpeg" in req)
    _check("req: pyobjc marked darwin-only", 'pyobjc-framework-AVFoundation; sys_platform == "darwin"' in req)
    for bad in ("tensorflow-macos", "tensorflow-metal", "uvloop"):
        _check(f"req: no {bad}", bad not in req)
    dev = _text("requirements-dev.txt")
    _check("req-dev: dmgbuild marked darwin-only", 'dmgbuild; sys_platform == "darwin"' in dev)
    _check("req-dev: plain 'dmgbuild' line absent (breaks pip on Windows)",
           not re.search(r"^dmgbuild\s*$", dev, flags=re.M))
    _check("req-dev: playwright present (SPA tests)", "playwright" in dev)
    _check("req-dev: python-docx present (lab manual)", "python-docx" in dev)


# ── 2. native dialog API + dispatch + cancel contract ─────────────────────
def _check_file_dialog() -> None:
    fd = _text("file_dialog.py")
    _check("file_dialog: exists with the three public pickers",
           all(f"def {f}" in fd for f in ("pick_open_file", "pick_save_file", "pick_folder")))
    _check("file_dialog: macOS dispatches on sys.platform", 'sys.platform == "darwin"' in fd)
    _check("file_dialog: Windows dispatches on os.name == 'nt'", 'os.name == "nt"' in fd)
    _check("file_dialog: native ctypes tier (IFileOpenDialog)", "IFileOpenDialog" in fd)
    _check("file_dialog: cancel is a silent None (no fallback after native)",
           "# canceled" in fd and "return None" in fd)
    _check("file_dialog: STA thread for COM", "STA" in fd or "COINIT_APARTMENTTHREADED" in fd)
    _check("file_dialog: pre-warm helper", "_warm_native_dialog" in fd)
    # Call sites must route through file_dialog, never raw osascript/tkinter.
    for mod in ("app.py", "record_controller.py"):
        src = _text(mod)
        _check(f"dialog call sites: {mod} uses file_dialog", "file_dialog" in src)
        _check(f"dialog call sites: {mod} has no osascript", "osascript" not in src)
        _check(f"dialog call sites: {mod} has no tkinter picker code",
               "import tkinter" not in src and "from tkinter" not in src)


# ── 3. explicit UTF-8 on every text-mode I/O ──────────────────────────────
def _check_encoding() -> None:
    for mod in ("app.py", "record_controller.py", "trainer.py", "dataset_io.py",
                "ocr_preprocess.py", "ocr_dataset.py", "ocr_trainer.py",
                "ui_styles.py", "image_preprocess.py", "serial_device.py",
                "camera_permission.py", "desktop_launcher.py"):
        try:
            tree = ast.parse(_text(mod), filename=mod)
        except SyntaxError as e:
            _check(f"encoding: {mod} parses", False, str(e))
            continue
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "open":
                    # Builtin open() only (skip Image.open / gzip.open / Path.open).
                    # Binary opens ("rb"/"wb") correctly carry no encoding.
                    mode = None
                    if len(node.args) >= 2:
                        mode = node.args[1]  # open(path, "rb")
                    for kw in node.keywords:
                        if kw.arg == "mode":
                            mode = kw.value
                    if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value:
                        continue
                    if "encoding" not in {kw.arg for kw in node.keywords}:
                        bad.append(f"line {node.lineno}: open() without encoding=")
                elif isinstance(fn, ast.Attribute) and fn.attr in ("read_text", "write_text"):
                    if "encoding" not in {kw.arg for kw in node.keywords}:
                        bad.append(f"line {node.lineno}: {fn.attr}() without encoding=")
        _check(f"encoding: {mod} — all text I/O explicit utf-8", not bad, "; ".join(bad[:3]))


# ── 4. camera + serial + subprocess/signals ───────────────────────────────
def _check_hardware() -> None:
    rc = _text("record_controller.py")
    _check("camera: CAP_DSHOW probe", "CAP_DSHOW" in rc)
    _check("camera: CAP_MSMF probe", "CAP_MSMF" in rc)
    _check("camera: dshow/msmf cold-start warmup retry", "cold-start warmup" in rc)
    _check("camera: Windows friendly-name enumeration", "_enumerate_windows_webcams" in rc)
    _check("camera: ffmpeg -list_devices dshow path", "-list_devices" in rc)
    _check("camera: 30s webcam cache TTL", "_WIN_WEBCAM_CACHE_TTL_S = 30.0" in rc)
    _check("camera: no-console subprocess kwargs", "CREATE_NO_WINDOW" in rc)
    _check("camera: pre-warm at server start", "_warm_windows_webcam_list" in rc and "_warm_windows_file_dialog" in rc)
    _check("camera: macOS bridge stays darwin-only", "sys.platform == \"darwin\"" in rc)
    sd = _text("serial_device.py")
    _check("serial: COM0-99 fallback enumeration", "_fallback_windows_com_ports" in sd)
    _check("serial: fallback wired into port listing", 'os.name == "nt"' in sd)
    for mod in ("ocr_preprocess.py", "ocr_dataset.py", "ocr_trainer.py", "app.py",
                "record_controller.py", "trainer.py", "desktop_launcher.py"):
        src = _text(mod)
        hit = None
        for bad in ("os.killpg(", "os.setsid(", "signal.SIGKILL", "import fcntl",
                    "import termios", "os.fork("):
            if bad in src:
                hit = bad
                break
        _check(f"subprocess/signal: {mod} free of POSIX-only calls",
               hit is None, f"found {hit}" if hit else "")


# ── 5. desktop launcher (WebView2 runtime) ────────────────────────────────
def _check_launcher() -> None:
    dl = _text("desktop_launcher.py")
    _check("launcher: no js_api on the non-darwin window (GIL deadlock)", "js_api=shell_api" in dl)
    _check("launcher: js_api only under darwin", "if sys.platform == \"darwin\":" in dl)
    _check("launcher: private_mode=False (WebView2 storage)", "private_mode=False" in dl)
    _check("launcher: chdir off the launch dir before spawning", "os.chdir(str(_data_dir))" in dl)
    _check("launcher: multiprocessing spawn guard for nt",
           'os.name == "nt"' in dl and 'set_start_method("spawn"' in dl)
    _check("launcher: freeze_support called", "multiprocessing.freeze_support()" in dl)
    _debug_post_noop = False
    try:
        tree = ast.parse(dl)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_debug_post":
                stmts = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
                _debug_post_noop = bool(stmts) and isinstance(stmts[0], ast.Return) and stmts[0].value is None
    except SyntaxError:
        _debug_post_noop = False
    _check("launcher: debug post is a hard no-op", "_debug_post" in dl and _debug_post_noop)
    _check("launcher: no resize/layout window event handlers (drag deadlock)",
           "window.events.resize" not in dl and "window.events.loaded" not in dl
           and "window.events.shown" not in dl and "window.events.maximized" not in dl)
    _check("launcher: splash reverted (dead code, no caller)",
           "_SPLASH_HTML" in dl and dl.count("_SPLASH_HTML") == 1)
    _check("launcher: shutdown terminate→kill→exit", "proc.terminate()" in dl and "proc.kill()" in dl)


# ── 6. SPA Windows layout fixes (embedded in app.py) ──────────────────────
def _check_spa() -> None:
    app = _text("app.py")
    _check("spa: fixed 1000px canvas", "min-width: 1000px" in app and "width: 1000px" in app)
    _check("spa: iframe mount keeps height=2000 + scrolling=True",
           "height=2000" in app and "scrolling=True" in app)
    _check("spa: reportLayoutDiag machinery", "function reportLayoutDiag" in app and "/diag" in app)
    _check("spa: measure/jitter failure diagnostics", "failed: 'measure'" in app and "failed: 'jitter'" in app)
    _check("spa: nudgeParentLayout parent resize kick",
           "function nudgeParentLayout" in app and "nudgeParentLayout();" in app)
    _check("spa: dbgEvent Windows latch + Mac url init",
           "if (!DEBUG_ENABLED || debugEventDisabled) return;" in app and "debug_server_url" in app)
    _check("spa: scrollSettingsIntoView kept", "function scrollSettingsIntoView" in app)
    _check("spa: settings open re-queries serial ports", "refreshSerialPorts(false, 'settingsSerialPort')" in app)
    _check("spa: no fluid full-bleed width regression", "max-width: min(1680px" not in app)
    _check("spa: no 100vh iframe height regression", "min-height: 100vh" not in app)


# ── 7. PyInstaller spec + packaging scripts ───────────────────────────────
def _check_packaging() -> None:
    spec = _text("TFLiteTraining.spec")
    _check("spec: file_dialog.py bundled", "file_dialog.py" in spec)
    _check("spec: TF include headers excluded (MAX_PATH)", "tensorflow/include" in spec)
    _check("spec: imageio_ffmpeg collected", "imageio_ffmpeg" in spec)
    _check("spec: pyserial windows hiddenimports", "serial.tools.list_ports_windows" in spec)
    _check("spec: multiprocessing hiddenimports", "multiprocessing.spawn" in spec)
    _check("spec: BUNDLE only on darwin", 'sys.platform == "darwin"' in spec)
    _check("spec: .ico icon for Windows", "AIoScouts.ico" in spec)
    for f in ("build_windows.ps1", "package_windows.ps1", "installer.iss"):
        _check(f"packaging: {f} present", (ROOT / f).exists())
    bw = _text("build_windows.ps1")
    _check("packaging: build pins Python 3.11", "py -3.11" in bw)
    _check("packaging: build reaps WebView2 before clean", "msedgewebview2" in bw)


# ── 8. file locking / cleanup patterns ────────────────────────────────────
def _check_cleanup() -> None:
    app = _text("app.py")
    _check("cleanup: async workspace rmtree (WinError 32)", "_remove_workspace_tree_async" in app)
    _check("cleanup: rmtree uses ignore_errors", "shutil.rmtree(" in app and "ignore_errors=True" in app)
    rc = _text("record_controller.py")
    _check("cleanup: probe-then-write with missing_ok unlink", "unlink(missing_ok=True)" in rc)


def main() -> None:
    _check_requirements()
    _check_file_dialog()
    _check_encoding()
    _check_hardware()
    _check_launcher()
    _check_spa()
    _check_packaging()
    _check_cleanup()
    print(f"\n{_PASSED} passed, {len(_FAILED)} failed")
    if _FAILED:
        print("\nFailed checks:")
        for name, detail in _FAILED:
            print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
