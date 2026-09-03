"""
FastAPI de la consola de operador, con un worker y el estado caliente en proceso. sirve la API
bajo /api y el frontend estatico (HTML, Alpine, uPlot, sin build) en /.

al arrancar calienta en segundo plano el contexto TrajDev de la granja C, sin bloquear el
arranque, que la primera peticion de /cases lo espera de todos modos. la granja B se calienta
a demanda.

  .venv/Scripts/uvicorn webapp.api.main:app        (navegador, http://127.0.0.1:8000)
  .venv/Scripts/python webapp/desktop.py           (ventana de escritorio, pywebview)
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import context as ctx
from . import jobs
from .routes import router

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def _prewarm_plan() -> None:
    """plan del caso estrella (C ev11) en un hilo aparte, que tarda ~2 min de NSGA-II derate
    y stop y no debe ocupar el worker unico. al terminar la demo por defecto abre al vuelo."""
    try:
        w = "era5" if ctx.era5_available() else "synthetic"
        ctx.plan_bundle("C", 11, "2026-10-01", True, w, 30, 42, 180, False, 10)
    except Exception:  # noqa: BLE001
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _warm():
        # calienta lo barato por el worker, contexto TrajDev de C y resumen mensual ERA5, que
        # retrasa segundos las primeras peticiones y no minutos. el plan estrella va aparte, y
        # un fallo aqui no puede tumbar el arranque
        try:
            await jobs.run(ctx.warm, "C", 42)
            await jobs.run(ctx.weather_month, 10)
        except Exception:  # noqa: BLE001
            pass
        threading.Thread(target=_prewarm_plan, daemon=True).start()
    task = asyncio.create_task(_warm())
    yield
    task.cancel()


app = FastAPI(title="Consola de operador — mantenimiento predictivo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")


# sonda de arranque del frontend, dice si hay era5 descargado y si ollama responde
@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "era5": ctx.era5_available(), "ollama": ctx.ollama_available()}


# el frontend estatico va despues del router, es un catch-all en / que sirve index.html
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
