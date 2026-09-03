"""
lanzador de escritorio, arranca la FastAPI local (uvicorn en un hilo) y abre la consola en una
ventana nativa (pywebview y WebView2). si esa ventana no llega a mostrarse cae al navegador,
para que el doble clic siempre muestre algo. el backend reutiliza src/ igual que en modo web.

  .venv/Scripts/python webapp/desktop.py           # ventana nativa, con fallback a navegador
  .venv/Scripts/python webapp/desktop.py --web      # solo servidor y navegador, sin ventana
  .venv/Scripts/uvicorn webapp.api.main:app         # solo servidor, http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# bajo pythonw.exe, sin consola, sys.stdout y sys.stderr son None y el logging de uvicorn
# los usa (llama a isatty), lo que tumba el arranque. se les da devnull antes de importarlo
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import uvicorn  # noqa: E402

HOST = "127.0.0.1"
WEB_ONLY = "--web" in sys.argv or "--no-window" in sys.argv


def _free_port() -> int:
    # el puerto 0 hace que el sistema asigne uno libre, y se lee antes de cerrar el socket
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_ready(port: int, timeout: float = 30.0) -> bool:
    """espera a que uvicorn acepte conexiones antes de abrir la ventana."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((HOST, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _serve_browser(url: str, server) -> None:
    """abre el navegador y mantiene vivo el servidor, el fallback fiable, sin WebView2."""
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.should_exit = True


def main() -> None:
    from webapp.api.main import app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    if not _wait_ready(port):
        return
    url = f"http://{HOST}:{port}/"

    if WEB_ONLY:
        _serve_browser(url, server)
        return

    # ventana nativa, y si algo falla (WebView2, pythonnet u otro) o no llega a mostrarse,
    # que se nota en que webview.start() retorna casi instantaneo, se cae al navegador
    try:
        import webview  # import perezoso, WebView2 solo hace falta para la ventana nativa
        webview.create_window("Mantenimiento predictivo - Consola de operador", url,
                              width=1220, height=1000, min_size=(900, 700))
        t0 = time.time()
        webview.start()
        if time.time() - t0 < 2.0:  # no llego a mostrarse, al navegador
            _serve_browser(url, server)
            return
    except Exception:  # noqa: BLE001
        _serve_browser(url, server)
        return
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
