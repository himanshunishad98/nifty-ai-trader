from __future__ import annotations

import ctypes
import json
import socket
import subprocess
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PORT = 8501
URL = f"http://localhost:{PORT}"
PID_FILE = ROOT / "data" / "streamlit.pid"
DASHBOARD = ROOT / "dashboard" / "dashboard.py"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BG = "#0a0f1c"
PANEL = "#101826"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
GREEN = "#22c55e"
RED = "#ff4d4f"
YELLOW = "#facc15"


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(json.loads(PID_FILE.read_text()).get("pid"))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _write_pid(pid: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps({"pid": pid}))


def _clear_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def _is_running(pid: int | None) -> bool:
    if not pid:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def running_pid() -> int | None:
    pid = _read_pid()
    if _is_running(pid):
        return pid
    _clear_pid()
    return None


def wait_for_server(timeout: int = 20) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        if _port_open():
            return True
        time.sleep(0.5)
    return False


def start_terminal() -> tuple[bool, str]:
    pid = running_pid()
    if pid:
        return True, f"Terminal already running on {URL}"
    if not PYTHON.exists():
        return False, f"Python not found at {PYTHON}"
    process = subprocess.Popen(
        [str(PYTHON), "-m", "streamlit", "run", str(DASHBOARD), "--server.headless", "true", "--server.port", str(PORT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    _write_pid(process.pid)
    if wait_for_server():
        return True, f"Terminal started on {URL}"
    return False, "Terminal started, but the web server did not respond yet."


def stop_terminal() -> tuple[bool, str]:
    pid = running_pid()
    if not pid:
        return True, "Terminal is not running."
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    time.sleep(1)
    _clear_pid()
    return True, "Terminal stopped."


def open_terminal() -> None:
    webbrowser.open(URL)


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NIFTY AI Trader Launcher")
        self.geometry("440x420")
        self.minsize(440, 420)
        self.resizable(False, False)
        self.configure(bg=BG)
        self.status_var = tk.StringVar()
        self.detail_var = tk.StringVar(value=URL)
        self._build()
        self.refresh_status()

    def _build(self) -> None:
        wrap = tk.Frame(self, bg=BG, padx=18, pady=18)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="NIFTY AI Trader", bg=BG, fg=TEXT, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(wrap, text="Start the trading terminal without using code.", bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 14))

        panel = tk.Frame(wrap, bg=PANEL, highlightthickness=1, highlightbackground="#1f2937", padx=14, pady=14)
        panel.pack(fill="x")
        tk.Label(panel, text="Status", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(panel, textvariable=self.status_var, bg=PANEL, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(6, 2))
        tk.Label(panel, textvariable=self.detail_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w")

        buttons = tk.Frame(wrap, bg=BG)
        buttons.pack(fill="x", pady=(16, 10))
        self._button(buttons, "Start Terminal", GREEN, self.start_and_open).pack(fill="x", pady=4)
        self._button(buttons, "Open Dashboard", YELLOW, self.open_dashboard).pack(fill="x", pady=4)
        self._button(buttons, "Stop Terminal", RED, self.stop_terminal).pack(fill="x", pady=4)

        footer = tk.Frame(wrap, bg=BG)
        footer.pack(fill="x", pady=(8, 0))
        tk.Label(footer, text="Dashboard URL", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(footer, text=URL, bg=BG, fg=TEXT, font=("Segoe UI", 10, "underline")).pack(anchor="w")

    def _button(self, parent: tk.Widget, text: str, color: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="#050816",
            activebackground=color,
            activeforeground="#050816",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            cursor="hand2",
        )

    def refresh_status(self) -> None:
        if running_pid() and _port_open():
            self.status_var.set("Running")
            self.detail_var.set(f"Live at {URL}")
        elif running_pid():
            self.status_var.set("Starting")
            self.detail_var.set("Process is running. Waiting for server.")
        else:
            self.status_var.set("Stopped")
            self.detail_var.set("Press Start Terminal to launch the app.")

    def start_and_open(self) -> None:
        ok, message = start_terminal()
        self.refresh_status()
        if ok:
            open_terminal()
        else:
            messagebox.showerror("Launcher", message)

    def open_dashboard(self) -> None:
        if not running_pid():
            ok, message = start_terminal()
            if not ok:
                messagebox.showerror("Launcher", message)
                return
        self.refresh_status()
        open_terminal()

    def stop_terminal(self) -> None:
        _, message = stop_terminal()
        self.refresh_status()
        messagebox.showinfo("Launcher", message)


if __name__ == "__main__":
    LauncherApp().mainloop()
