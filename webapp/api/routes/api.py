"""
endpoints de la consola, finos, resuelven los parametros por defecto y delegan el trabajo
pesado (torch, NSGA-II) al worker unico (jobs.run) para no bloquear el event-loop.

los calculos son idempotentes, no cambian estado, y por eso se sirven por GET. solo /chat es
POST, que lleva la pregunta del operador y responde en streaming, el efecto de terminal CRT.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import context as ctx
from .. import jobs
from .. import llm_stream

router = APIRouter()


def _resolve(farm: str, event: int, seed: int, mes: int | None, weather: str | None) -> tuple:
    """resuelve el mes (por defecto el real de la alarma), la meteo (era5 si esta descargado)
    y t0."""
    if mes is None:
        mes = ctx.alarm_month(farm, event, seed) or 6
    weather = weather or ("era5" if ctx.era5_available() else "synthetic")
    return mes, weather, date(2026, mes, 1).isoformat()


# seleccion
@router.get("/farms")
async def farms() -> dict:
    return {"farms": [
        {"id": "C", "nombre": "Farm C", "n_turbinas": ctx.N_TURBINAS["C"], "sensores": 957},
        {"id": "B", "nombre": "Farm B", "n_turbinas": ctx.N_TURBINAS["B"], "sensores": 257},
    ], "era5": ctx.era5_available(), "ollama": ctx.ollama_available()}


@router.get("/farms/{farm}/cases")
async def cases(farm: str, seed: int = 42) -> dict:
    _check_farm(farm)
    await jobs.run(ctx.warm, farm, seed)
    return {"farm": farm, "cases": await jobs.run(ctx.list_cases, farm, seed)}


# reproduccion
@router.get("/cases/{farm}/{event}/playback")
async def playback(farm: str, event: int, seed: int = 42) -> dict:
    _check_farm(farm)
    await jobs.run(ctx.warm, farm, seed)
    return await jobs.run(ctx.playback, farm, event, seed)


# diagnostico (Agente 1)
@router.get("/cases/{farm}/{event}/diagnose")
async def diagnose(farm: str, event: int, seed: int = 42, use_llm: bool = False) -> dict:
    _check_farm(farm)
    return await jobs.run(ctx.diagnose_bundle, farm, event, seed, use_llm)


# plan (Agente 2 y solver)
# si el triaje no escala no hay plan que resolver, se devuelve el motivo del triaje
@router.get("/cases/{farm}/{event}/plan")
async def plan(farm: str, event: int, seed: int = 42, use_llm: bool = False,
               con: bool = True, weather: str | None = None, nsc: int = 30,
               horizonte: int = 180, mes: int | None = Query(default=None)) -> dict:
    _check_farm(farm)
    mes, weather, t0 = _resolve(farm, event, seed, mes, weather)
    res = await jobs.run(ctx.plan_bundle, farm, event, t0, con, weather, nsc, seed,
                         horizonte, use_llm, mes)
    if res is None:
        rep = await jobs.run(ctx.diagnose, farm, event, seed, False)
        return {"escalado": False, "triage": rep.triage.action.value,
                "razones": rep.triage.razones}
    return res


# verdad, solo tras decidir
@router.get("/cases/{farm}/{event}/truth")
async def truth(farm: str, event: int, seed: int = 42) -> dict:
    _check_farm(farm)
    return await jobs.run(ctx.truth, farm, event, seed)


# terminal LLM (streaming)
class ChatRequest(BaseModel):
    farm: str
    event: int
    seed: int = 42
    question: str
    con: bool = True
    weather: str | None = None
    nsc: int = 30
    horizonte: int = 180
    mes: int | None = None


# fuera de ambito se responde 200 con la frase fija, el cliente lo distingue por X-Scope
@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    _check_farm(req.farm)
    if not ctx.ollama_available(refresh=True):
        raise HTTPException(503, "Ollama no disponible (arranca 'ollama serve' y el modelo qwen3).")
    dentro = await jobs.run(llm_stream.in_scope, req.question)
    # sin buffering intermedio, cada token tiene que llegar segun se genera
    headers = {"X-Scope": "in" if dentro else "out", "Cache-Control": "no-cache",
               "X-Accel-Buffering": "no"}
    if not dentro:
        def _oos():
            yield llm_stream.OUT_OF_SCOPE
        return StreamingResponse(_oos(), media_type="text/plain; charset=utf-8", headers=headers)

    mes, weather, t0 = _resolve(req.farm, req.event, req.seed, req.mes, req.weather)
    contexto = await jobs.run(llm_stream.build_context, req.farm, req.event, req.seed, t0,
                              req.con, weather, req.nsc, req.horizonte, mes)

    def _gen():
        yield from llm_stream.stream_answer(contexto, req.question)
    return StreamingResponse(_gen(), media_type="text/plain; charset=utf-8", headers=headers)


def _check_farm(farm: str) -> None:
    if farm not in ctx.N_TURBINAS:
        raise HTTPException(404, f"granja desconocida: {farm}")
