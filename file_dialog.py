from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows file dialogs, in order of preference:
#   1. NATIVE common dialogs via ctypes (IFileOpenDialog / IFileSaveDialog) —
#      zero subprocesses, the dialog opens instantly (the macOS-osascript
#      experience).  Runs on a dedicated STA thread so COM apartment rules
#      can never bite (RPC_E_CHANGED_MODE etc.).
#   2. PowerShell + System.Windows.Forms (the previous implementation) —
#      works but costs ~1 s of PowerShell startup and needs CREATE_NO_WINDOW.
#   3. tkinter — last resort.
# macOS uses osascript throughout (instant, native).
# Cancel semantics are threaded through: a user CANCEL on any tier returns
# None and NEVER opens the next tier's dialog.


# --------------------------------------------------------------------------
# Native Windows dialogs (ctypes) — reviewed against the Windows SDK
# --------------------------------------------------------------------------

def _win_native_dialog(
    *,
    title: str = "",
    default_name: str = "",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    initial_dir: Optional[str] = None,
    pick_folder: bool = False,
    save: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Show the native Windows common file dialog.

    Returns ``(True, path)`` on success, ``(False, None)`` when the user
    canceled, and raises on failure (the caller then falls back to the
    PowerShell dialog).  Never call this on non-Windows.
    """
    result: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["out"] = _win_native_dialog_sta(
                title=title,
                default_name=default_name,
                filetypes=filetypes,
                initial_dir=initial_dir,
                pick_folder=pick_folder,
                save=save,
            )
        except Exception as e:  # propagate to the caller thread
            result["err"] = e

    # Dedicated thread: COM initializes per-thread and a thread that was
    # already initialized MTA would reject STA (RPC_E_CHANGED_MODE) — a
    # fresh thread can never hit that, and Show() pumps messages itself
    # while we join().
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "err" in result:
        raise result["err"]
    return result.get("out")


def _win_native_dialog_sta(
    *,
    title: str,
    default_name: str,
    filetypes: Optional[List[Tuple[str, str]]],
    initial_dir: Optional[str],
    pick_folder: bool,
    save: bool,
) -> Tuple[bool, Optional[str]]:
    import ctypes
    from ctypes import wintypes

    ole32 = ctypes.WinDLL("ole32")
    shell32 = ctypes.WinDLL("shell32")

    HRESULT = ctypes.c_long

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", wintypes.BYTE * 8),
        ]

    assert ctypes.sizeof(GUID) == 16, "GUID must be 16 bytes (Windows packing)"

    CLSID_FileOpenDialog = GUID(0xDC1C5A9C, 0xE88A, 0x4DDE, (0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7))
    CLSID_FileSaveDialog = GUID(0xC0B4E2F3, 0xBA21, 0x4773, (0x8D, 0xBA, 0x33, 0x5E, 0xC9, 0x46, 0xEB, 0x8B))
    IID_IFileOpenDialog = GUID(0xD57C7288, 0xD4AD, 0x4768, (0xBE, 0x02, 0x9D, 0x96, 0x95, 0x32, 0xD9, 0x60))
    IID_IFileSaveDialog = GUID(0x84BCCD23, 0x5FDE, 0x4CDB, (0xAE, 0xA4, 0xAF, 0x64, 0xB8, 0x3D, 0x78, 0xAB))
    IID_IShellItem = GUID(0x43826D1E, 0xE718, 0x42EE, (0xBC, 0x55, 0xA1, 0xE2, 0x61, 0xC3, 0x7B, 0xFE))

    CLSCTX_INPROC_SERVER = 0x1
    COINIT_APARTMENTTHREADED = 0x2
    SIGDN_FILESYSPATH = 0x80058000
    HR_ERROR_CANCELLED = -2147023673  # 0x800704C7 (HRESULT_FROM_WIN32(ERROR_CANCELLED))

    FOS_OVERWRITEPROMPT = 0x2
    FOS_PICKFOLDERS = 0x20
    FOS_FORCEFILESYSTEM = 0x40
    FOS_PATHMUSTEXIST = 0x800
    FOS_FILEMUSTEXIST = 0x1000
    FOS_CREATEPROMPT = 0x2000
    FOS_NOREADONLYRETURN = 0x8000
    FOS_DONTADDTORECENT = 0x2000000

    # IFileDialog vtable slots (IUnknown 0-2, IModalWindow 3) — verified
    # against shobjidl.h / win32metadata:
    IDX_SHOW = 3
    IDX_SET_FILETYPES = 4
    IDX_SET_FILETYPE_INDEX = 5
    IDX_SET_OPTIONS = 9
    IDX_GET_OPTIONS = 10
    IDX_SET_DEFAULT_FOLDER = 11
    IDX_SET_FILENAME = 15
    IDX_SET_TITLE = 17
    IDX_GET_RESULT = 20
    IDX_SET_DEFAULT_EXTENSION = 22
    # IShellItem vtable (IUnknown 0-2, 3 BindToHandler, 4 GetParent):
    IDX_GET_DISPLAY_NAME = 5
    # IUnknown:
    IDX_RELEASE = 2

    class FILTERSPEC(ctypes.Structure):
        _fields_ = [("pszName", wintypes.LPCWSTR), ("pszSpec", wintypes.LPCWSTR)]

    # Flat API prototypes (argtypes/restype set explicitly — untyped ctypes
    # calls truncate pointers to 32-bit c_int).
    ole32.CoInitializeEx.restype = HRESULT
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoUninitialize.restype = None
    ole32.CoUninitialize.argtypes = []
    ole32.CoTaskMemFree.restype = None
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoCreateInstance.restype = HRESULT
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(GUID), ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p),
    ]
    shell32.SHCreateItemFromParsingName.restype = HRESULT
    shell32.SHCreateItemFromParsingName.argtypes = [
        wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p),
    ]

    def _method(ptr, index, restype, *argtypes):
        # vtbl[index] is a plain int function address — pass the INT (a
        # c_void_p instance here raises TypeError in the ctypes
        # function-pointer constructor).
        vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
        return proto(vtbl[index])

    def _release(ptr) -> None:
        if ptr:
            try:
                _method(ptr, IDX_RELEASE, wintypes.ULONG)(ptr)
            except Exception:
                pass

    hres = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    if hres < 0:
        # S_OK(0)/S_FALSE(1) both count as initialized; anything negative
        # (e.g. RPC_E_CHANGED_MODE) is a hard error here because this is
        # already the dedicated STA thread.
        raise OSError(f"CoInitializeEx failed: 0x{hres & 0xFFFFFFFF:08X}")
    dialog_ptr = None
    folder_item = None
    try:
        clsid = CLSID_FileSaveDialog if save else CLSID_FileOpenDialog
        iid = IID_IFileSaveDialog if save else IID_IFileOpenDialog
        dialog = ctypes.c_void_p()
        hres = ole32.CoCreateInstance(
            ctypes.byref(clsid), None, CLSCTX_INPROC_SERVER, ctypes.byref(iid), ctypes.byref(dialog)
        )
        if hres < 0 or not dialog.value:
            raise OSError(f"CoCreateInstance failed: 0x{hres & 0xFFFFFFFF:08X}")
        dialog_ptr = dialog.value

        # Merge our flags into the dialog's OWN defaults (GetOptions →
        # OR → SetOptions); SetOptions REPLACES the whole mask.
        cur_opts = wintypes.DWORD()
        hres = _method(dialog_ptr, IDX_GET_OPTIONS, HRESULT, ctypes.POINTER(wintypes.DWORD))(
            dialog_ptr, ctypes.byref(cur_opts)
        )
        if hres < 0:
            raise OSError(f"GetOptions failed: 0x{hres & 0xFFFFFFFF:08X}")
        options = cur_opts.value | FOS_FORCEFILESYSTEM | FOS_DONTADDTORECENT
        if pick_folder:
            options |= FOS_PICKFOLDERS | FOS_PATHMUSTEXIST
        elif save:
            options |= FOS_OVERWRITEPROMPT | FOS_PATHMUSTEXIST | FOS_CREATEPROMPT
        else:
            options |= FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST | FOS_NOREADONLYRETURN
        hres = _method(dialog_ptr, IDX_SET_OPTIONS, HRESULT, wintypes.UINT)(dialog_ptr, options)
        if hres < 0:
            raise OSError(f"SetOptions failed: 0x{hres & 0xFFFFFFFF:08X}")

        if title:
            _method(dialog_ptr, IDX_SET_TITLE, HRESULT, wintypes.LPCWSTR)(dialog_ptr, title)

        if default_name:
            _method(dialog_ptr, IDX_SET_FILENAME, HRESULT, wintypes.LPCWSTR)(dialog_ptr, default_name)
            if save:
                ext = Path(default_name).suffix or ""
                if ext:
                    # SetDefaultExtension takes the extension WITHOUT the dot.
                    _method(dialog_ptr, IDX_SET_DEFAULT_EXTENSION, HRESULT, wintypes.LPCWSTR)(
                        dialog_ptr, ext.lstrip(".")
                    )

        if filetypes and not pick_folder:
            # Append "All files" so non-matching files stay selectable (the
            # PowerShell/tkinter tiers do the same).
            specs_list = list(filetypes) + [("All files", "*.*")]
            specs = (FILTERSPEC * len(specs_list))()
            for i, (label, pattern) in enumerate(specs_list):
                specs[i].pszName = label
                specs[i].pszSpec = pattern
            hres = _method(dialog_ptr, IDX_SET_FILETYPES, HRESULT, wintypes.UINT, ctypes.c_void_p)(
                dialog_ptr, len(specs_list), ctypes.cast(specs, ctypes.c_void_p)
            )
            if hres < 0:
                raise OSError(f"SetFileTypes failed: 0x{hres & 0xFFFFFFFF:08X}")
            _keep_alive = specs  # noqa: F841 — must outlive Show()
            _method(dialog_ptr, IDX_SET_FILETYPE_INDEX, HRESULT, wintypes.UINT)(dialog_ptr, 1)

        if initial_dir:
            folder_item = ctypes.c_void_p()
            hres = shell32.SHCreateItemFromParsingName(
                str(Path(initial_dir).expanduser().resolve()),
                None,
                ctypes.byref(IID_IShellItem),
                ctypes.byref(folder_item),
            )
            # Non-existent path → silently skip the default folder.
            if hres >= 0 and folder_item.value:
                _method(dialog_ptr, IDX_SET_DEFAULT_FOLDER, HRESULT, ctypes.c_void_p)(
                    dialog_ptr, folder_item.value
                )

        hres = _method(dialog_ptr, IDX_SHOW, HRESULT, wintypes.HWND)(dialog_ptr, None)
        if hres == HR_ERROR_CANCELLED:
            return False, None
        if hres < 0:
            raise OSError(f"Show failed: 0x{hres & 0xFFFFFFFF:08X}")

        item = ctypes.c_void_p()
        hres = _method(dialog_ptr, IDX_GET_RESULT, HRESULT, ctypes.POINTER(ctypes.c_void_p))(
            dialog_ptr, ctypes.byref(item)
        )
        if hres < 0 or not item.value:
            raise OSError(f"GetResult failed: 0x{hres & 0xFFFFFFFF:08X}")
        try:
            name = wintypes.LPWSTR()
            hres = _method(item.value, IDX_GET_DISPLAY_NAME, HRESULT, wintypes.UINT, ctypes.POINTER(wintypes.LPWSTR))(
                item.value, SIGDN_FILESYSPATH, ctypes.byref(name)
            )
            try:
                if hres < 0 or not name.value:
                    raise OSError(f"GetDisplayName failed: 0x{hres & 0xFFFFFFFF:08X}")
                return True, str(name.value)
            finally:
                ole32.CoTaskMemFree(name)  # NULL-safe
        finally:
            _release(item.value)
    finally:
        if folder_item and folder_item.value:
            _release(folder_item.value)
        if dialog_ptr:
            _release(dialog_ptr)
        ole32.CoUninitialize()


# --------------------------------------------------------------------------
# PowerShell fallback dialogs
# --------------------------------------------------------------------------

def _pick_via_powershell(script_body: str) -> Tuple[bool, Optional[str]]:
    """Run a PowerShell dialog.

    Returns ``(True, path)`` on success, ``(False, None)`` when the user
    canceled, and raises on failure (PowerShell could not run at all).

    ``script_body`` must set a ``$result`` variable (empty string means
    canceled).
    """
    out_path = Path(tempfile.gettempdir()) / f"tfldlg_{uuid.uuid4().hex}.txt"
    script = (
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null; "
        "$result = ''; "
        + script_body
        + f"; $result | Out-File -FilePath '{out_path}' -Encoding utf8"
    )
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            timeout=120,
            creationflags=flags,
        )
    except Exception as e:
        raise RuntimeError(f"powershell dialog failed to run: {e}") from e
    try:
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8-sig").strip()
            return (bool(text), text or None)
        raise RuntimeError(f"powershell dialog failed (exit {proc.returncode})")
    except Exception:
        raise
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass


def _quote_ps(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _ps_filter_str(filetypes: Optional[List[Tuple[str, str]]]) -> str:
    filter_parts: List[str] = []
    if filetypes:
        for label, pattern in filetypes:
            filter_parts.append(f"{label} ({pattern})|{pattern}")
    if filter_parts:
        filter_parts.append("All files (*.*)|*.*")
    return "|".join(filter_parts)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def pick_open_file(
    title: str = "Open",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    if sys.platform == "darwin":
        try:
            script = f'POSIX path of (choose file with prompt "{title}")'
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                picked = (proc.stdout or "").strip()
                if picked:
                    return picked
        except Exception:
            pass
    if os.name == "nt":
        try:
            ok, path = _win_native_dialog(title=title, filetypes=filetypes, initial_dir=initial_dir)
            if ok:
                return path
            return None  # user canceled — never open a second dialog
        except Exception:
            pass  # native failed — fall through to PowerShell
        body = (
            f"$d = New-Object System.Windows.Forms.OpenFileDialog; "
            f"$d.Title = {_quote_ps(title)}; "
            f"$d.Filter = {_quote_ps(_ps_filter_str(filetypes))}; "
            f"$d.FilterIndex = 1; "
        )
        if initial_dir:
            body += f"$d.InitialDirectory = {_quote_ps(initial_dir)}; "
        body += "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $result = $d.FileName }"
        try:
            ok, picked = _pick_via_powershell(body)
            if ok:
                return picked
            return None  # canceled
        except Exception:
            pass  # PowerShell failed — fall through to tkinter
    return _pick_open_file_tk(title, filetypes, initial_dir)


def pick_save_file(
    title: str = "Save",
    default_name: str = "project.tmproj",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    initial_dir: Optional[str] = None,
) -> Optional[str]:
    if sys.platform == "darwin":
        try:
            script = f'POSIX path of (choose file name with prompt "{title}" default name "{default_name}")'
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                picked = (proc.stdout or "").strip()
                if picked:
                    return picked
        except Exception:
            pass
    if os.name == "nt":
        try:
            ok, path = _win_native_dialog(
                title=title, default_name=default_name, filetypes=filetypes, initial_dir=initial_dir, save=True
            )
            if ok:
                return path
            return None  # canceled
        except Exception:
            pass
        body = (
            f"$d = New-Object System.Windows.Forms.SaveFileDialog; "
            f"$d.Title = {_quote_ps(title)}; "
            f"$d.FileName = {_quote_ps(default_name)}; "
            f"$d.Filter = {_quote_ps(_ps_filter_str(filetypes))}; "
            f"$d.AddExtension = $true; "
        )
        if initial_dir:
            body += f"$d.InitialDirectory = {_quote_ps(initial_dir)}; "
        body += "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $result = $d.FileName }"
        try:
            ok, picked = _pick_via_powershell(body)
            if ok:
                return picked
            return None  # canceled
        except Exception:
            pass
    return _pick_save_file_tk(title, default_name, filetypes, initial_dir)


def pick_folder(title: str = "Choose Folder", initial_dir: Optional[str] = None) -> Optional[str]:
    if sys.platform == "darwin":
        try:
            if initial_dir:
                start_posix = (
                    str(Path(initial_dir).expanduser().resolve()).replace("\\", "\\\\").replace('"', '\\"')
                )
                script = f'POSIX path of (choose folder with prompt "{title}" default location (POSIX file "{start_posix}"))'
            else:
                script = f'POSIX path of (choose folder with prompt "{title}")'
            proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                picked = (proc.stdout or "").strip()
                if picked:
                    return picked
        except Exception:
            pass
    if os.name == "nt":
        try:
            ok, path = _win_native_dialog(title=title, initial_dir=initial_dir, pick_folder=True)
            if ok:
                return path
            return None  # canceled
        except Exception:
            pass
        body = (
            f"$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$d.Description = {_quote_ps(title)}; "
            f"$d.ShowNewFolderButton = $true; "
        )
        if initial_dir:
            body += f"$d.SelectedPath = {_quote_ps(initial_dir)}; "
        body += "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $result = $d.SelectedPath }"
        try:
            ok, picked = _pick_via_powershell(body)
            if ok:
                return picked
            return None  # canceled
        except Exception:
            pass
    return _pick_folder_tk(title, initial_dir)


# --------------------------------------------------------------------------
# tkinter last-resort fallbacks
# --------------------------------------------------------------------------

def _pick_open_file_tk(
    title: str, filetypes: Optional[List[Tuple[str, str]]], initial_dir: Optional[str]
) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        return filedialog.askopenfilename(title=title, initialdir=initial_dir, filetypes=filetypes) or None
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _pick_save_file_tk(
    title: str, default_name: str, filetypes: Optional[List[Tuple[str, str]]], initial_dir: Optional[str]
) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        return (
            filedialog.asksaveasfilename(
                title=title,
                initialdir=initial_dir,
                initialfile=default_name,
                defaultextension=".tmproj",
                filetypes=filetypes,
            )
            or None
        )
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _pick_folder_tk(title: str, initial_dir: Optional[str]) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        return filedialog.askdirectory(title=title, initialdir=initial_dir) or None
    finally:
        try:
            root.destroy()
        except Exception:
            pass
