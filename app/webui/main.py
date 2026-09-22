from multiprocessing import freeze_support
import os
import sys
from pathlib import Path

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import webview

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def _show_startup_error(message: str):
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Konfigurationsfehler", message)


def main():
    try:
        from config.games import GAMES  # loest Validierung von games.json aus
    except Exception as e:
        _show_startup_error(str(e))
        raise SystemExit(1)

    from app.webui.api import Api

    api = Api()
    window = webview.create_window(
        "Highlight Cutter",
        str(FRONTEND_DIR / "index.html"),
        js_api=api,
        width=1320,
        height=880,
        min_size=(1100, 720),
    )
    api.window = window
    webview.start()


if __name__ == "__main__":
    freeze_support()  # wichtig fuer Windows + YOLO/Multiprocessing
    main()
