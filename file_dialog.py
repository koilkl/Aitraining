from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

# Windows-native dialogs via PowerShell + System.Windows.Forms. This avoids the
# tkinter-in-a-daemon-thread problem that breaks the Open/Save Project buttons in
# frozen Windows builds, and needs no extra runtime to be bundled.


def _pick_via_powershell(script_body: str) -> Optional[str]:
    """Run a PowerShell dialog and return the selected path, or None if canceled.

    ``script_body`` must set a ``$result`` variable (empty string means canceled).
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
        subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            timeout=120,
            creationflags=flags,
        )
    except Exception:
        return None
    try:
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8-sig").strip()
            return text or None
        return None
    except Exception:
        return None
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass


def _quote_ps(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


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
        filter_parts: List[str] = []
        if filetypes:
            for label, pattern in filetypes:
                filter_parts.append(f"{label} ({pattern})|{pattern}")
        if filter_parts:
            filter_parts.append("All files (*.*)|*.*")
        filter_str = "|".join(filter_parts)
        body = (
            f"$d = New-Object System.Windows.Forms.OpenFileDialog; "
            f"$d.Title = {_quote_ps(title)}; "
            f"$d.Filter = {_quote_ps(filter_str)}; "
            f"$d.FilterIndex = 1; "
        )
        if initial_dir:
            body += f"$d.InitialDirectory = {_quote_ps(initial_dir)}; "
        body += "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $result = $d.FileName }"
        picked = _pick_via_powershell(body)
        if picked:
            return picked
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
        filter_parts: List[str] = []
        if filetypes:
            for label, pattern in filetypes:
                filter_parts.append(f"{label} ({pattern})|{pattern}")
        if filter_parts:
            filter_parts.append("All files (*.*)|*.*")
        filter_str = "|".join(filter_parts)
        body = (
            f"$d = New-Object System.Windows.Forms.SaveFileDialog; "
            f"$d.Title = {_quote_ps(title)}; "
            f"$d.FileName = {_quote_ps(default_name)}; "
            f"$d.Filter = {_quote_ps(filter_str)}; "
            f"$d.AddExtension = $true; "
        )
        if initial_dir:
            body += f"$d.InitialDirectory = {_quote_ps(initial_dir)}; "
        body += "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $result = $d.FileName }"
        picked = _pick_via_powershell(body)
        if picked:
            return picked
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
        body = (
            f"$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$d.Description = {_quote_ps(title)}; "
            f"$d.ShowNewFolderButton = $true; "
        )
        if initial_dir:
            body += f"$d.SelectedPath = {_quote_ps(initial_dir)}; "
        body += "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $result = $d.SelectedPath }"
        picked = _pick_via_powershell(body)
        if picked:
            return picked
    return _pick_folder_tk(title, initial_dir)


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
