"""
catalogo citable de parametros de tareas, tool del Agente 2

carga src/scheduler/kb/tareas.yaml y construye una `MaintenanceTask` a partir de una entrada.
el planificador solo elige el subsistema y la ventana; la duracion, el coste y el riesgo vienen
del catalogo y nunca del LLM. los costes son relativos, porque el dataset CARE esta anonimizado
y normalizado.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.agents.contracts import MaintenanceTask, Recursos

CATALOG_PATH = Path(__file__).resolve().parents[2] / "scheduler" / "kb" / "tareas.yaml"

_CACHE: dict | None = None


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    """catalogo crudo, cacheado para la ruta por defecto."""

    global _CACHE
    # solo se cachea la ruta por defecto, una ruta a mano puede ser un catalogo de prueba
    if path == CATALOG_PATH:
        if _CACHE is None:
            _CACHE = yaml.safe_load(path.read_text(encoding="utf-8"))
        return _CACHE
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def subsistemas_catalogo(cat: dict | None = None) -> list[str]:
    return list((cat or load_catalog())["subsistemas"].keys())


def coste_generacion_dia(cat: dict | None = None) -> float:
    return float((cat or load_catalog())["globales"]["lucro_cesante"]["coste_generacion_dia"])


def recursos_catalogo(cat: dict | None = None, **kw) -> Recursos:
    """Recursos con la tarifa CTV del catalogo, los demas defaults desde el contrato."""

    cat = cat or load_catalog()
    return Recursos(coste_dia_ctv=float(cat["globales"]["recursos"]["coste_dia_ctv"]), **kw)


def _operacion_degradada(cat: dict) -> tuple[float, float, float]:
    """(derate_factor, p_fallo_max, factor_ventana_derating) del bloque globales del
    refinamiento de coste."""

    g = cat["globales"]["operacion_degradada"]
    return (float(g["derate_factor"]), float(g["p_fallo_max"]), float(g["factor_ventana_derating"]))


def _espera_emergencia(cat: dict, requiere_jackup: bool) -> int:
    """dias extra de parada de una reparacion de emergencia (jack-up ~45, CTV ~14)."""

    g = cat["globales"]["reaccion_emergencia"]
    return int(g["espera_emergencia_jackup_dias" if requiere_jackup else "espera_emergencia_ctv_dias"])


def _planificacion(cat: dict, requiere_jackup: bool) -> tuple[int, float]:
    """(lead_planificacion_dias, prima_urgencia_max) segun el recurso (jack-up ~30, CTV ~7)."""

    g = cat["globales"]["planificacion"]
    lead = int(g["lead_planificacion_jackup_dias" if requiere_jackup else "lead_planificacion_ctv_dias"])
    return lead, float(g["prima_urgencia_max"])


def _impacto_intervencion(cat: dict, requiere_jackup: bool) -> tuple[float, float]:
    """(factor_ruido_buque, factor_trafico_buque) segun el buque, jack-up ruidoso o CTV
    ligero."""

    g = cat["globales"]["impacto_intervencion"]
    suf = "jackup" if requiere_jackup else "ctv"
    return float(g[f"factor_ruido_{suf}"]), float(g[f"factor_trafico_{suf}"])


def catalog_task(subsistema: str, task_id: str, farm: str, asset_id: int, ventana_fin_dia: int,
                 origen_event_id: int | None = None, ventana_inicio_dia: int = 0,
                 cat: dict | None = None) -> MaintenanceTask:
    """`MaintenanceTask` desde una entrada del catalogo, con 'generico' si el subsistema no
    esta. ventana_fin_dia es el ultimo dia de inicio permitido y sale de la urgencia del
    diagnostico."""

    cat = cat or load_catalog()
    subs = cat["subsistemas"]
    key = subsistema if subsistema in subs else "generico"
    p = subs[key]
    # el buque decide de golpe la espera de emergencia, el lead y el impacto ambiental
    jackup = bool(p["requiere_jackup"])
    derate, p_fallo_max, factor_vent = _operacion_degradada(cat)
    lead, prima = _planificacion(cat, jackup)
    f_ruido, f_trafico = _impacto_intervencion(cat, jackup)
    return MaintenanceTask(
        task_id=task_id, farm=farm, asset_id=asset_id, origen_event_id=origen_event_id,
        subsistema=key, duracion_dias=int(p["duracion_dias"]),
        ventana_inicio_dia=ventana_inicio_dia,
        # el max evita una ventana invertida si la urgencia pide un fin anterior al inicio
        ventana_fin_dia=max(ventana_inicio_dia, int(ventana_fin_dia)),
        coste_intervencion=float(p["coste_intervencion"]),
        riesgo_fallo_base=float(p["riesgo_fallo_base"]),
        riesgo_crecimiento_dia=float(p["riesgo_crecimiento_dia"]),
        coste_generacion_dia=coste_generacion_dia(cat),
        coste_correctivo=float(p["coste_correctivo"]),
        requiere_jackup=jackup,
        derate_factor=derate, p_fallo_max=p_fallo_max, factor_ventana_derating=factor_vent,
        espera_emergencia_dias=_espera_emergencia(cat, jackup),
        lead_planificacion_dias=lead, prima_urgencia_max=prima,
        factor_ruido_buque=f_ruido, factor_trafico_buque=f_trafico)
