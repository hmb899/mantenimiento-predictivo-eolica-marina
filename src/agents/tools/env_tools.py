"""tools ambientales del Agente 2, consulta de la KB citable."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from src.agents.contracts import EnvPeriod, NoiseRule, VetoDuro

KB_PATH = Path(__file__).resolve().parents[2] / "envmod" / "kb" / "especies.yaml"


def load_kb(path: Path = KB_PATH) -> tuple[list[EnvPeriod], list[NoiseRule], dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # el yaml se valida al construir los contratos, un campo de mas o de menos revienta aqui
    periodos = [EnvPeriod(**p) for p in raw["periodos"]]
    reglas = [NoiseRule(**r) for r in raw["reglas_ruido"]]
    return periodos, reglas, raw["emision_operacional"]


def _month_in(mes: int, ini: int, fin: int) -> bool:
    # rangos que cruzan ano
    return ini <= mes <= fin if ini <= fin else (mes >= ini or mes <= fin)


def periodos_activos(dia: date, periodos: list[EnvPeriod] | None = None) -> list[EnvPeriod]:
    """periodos criticos activos en una fecha, tool del Agente 2 y del objetivo ambiental."""

    if periodos is None:
        periodos, _, _ = load_kb()
    return [p for p in periodos if _month_in(dia.month, p.mes_inicio, p.mes_fin)]


def sensibilidad_dia(dia: date, canal: str, periodos: list[EnvPeriod] | None = None) -> float:
    """peso ambiental de un dia para un canal de impacto, 'trafico_embarcaciones' o
    'ruido_operacional', el maximo de las sensibilidades activas que afectan a ese canal."""

    # el canal 'ambos' cuenta para los dos, por eso entra en las dos consultas
    activos = [p for p in periodos_activos(dia, periodos)
               if p.canal_impacto in (canal, "ambos")]
    # sin periodo activo el dia no penaliza, de ahi el default a cero
    return max((p.sensibilidad for p in activos), default=0.0)


def veto_jackup_periodo_sensible(periodos: list[EnvPeriod] | None = None) -> list[VetoDuro]:
    """veto duro de politica, no hardcodeado, que prohibe iniciar operaciones con jack-up, las
    mas ruidosas e intrusivas, durante el periodo de mayor sensibilidad acustica de la KB. se
    deriva del propio calendario, tomando el periodo de mayor sensibilidad que afecta al ruido
    operacional, la reproduccion de la marsopa, jun-ago 0.9. el resto de la fauna entra por el
    objetivo blando, la sensibilidad en f3, y no por veto."""

    if periodos is None:
        periodos, _, _ = load_kb()
    ruido = [p for p in periodos if p.canal_impacto in ("ruido_operacional", "ambos")]
    if not ruido:
        return []
    top = max(ruido, key=lambda p: p.sensibilidad)
    return [VetoDuro(alcance="jackup", mes_inicio=top.mes_inicio, mes_fin=top.mes_fin,
                     motivo=f"{top.actividad} {top.especie} (sensibilidad {top.sensibilidad}, KB)")]
