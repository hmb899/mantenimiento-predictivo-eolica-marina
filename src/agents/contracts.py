"""
contratos del sistema de agentes (pydantic v2), la fuente de verdad de las interfaces

los agentes, las tools y el scheduler solo se comunican con estos tipos, y toda salida de LLM
se valida contra ellos, nada entra al pipeline sin validar

flujo
    AlarmEvidence, AlarmReport (con TriageDecision), MaintenanceTask, PlanRequest, ParetoFront
    y MaintenancePlan
diseno de la fase
    el LLM percibe, traduce y explica, los numeros salen de las tools (src/xai, src/envmod,
    src/scheduler) y el plan lo produce NSGA-II
regla transversal
    la evidencia es label-free, la verdad-terreno CARE vive solo en EvalContext y jamas se
    serializa en un prompt (la eval de triaje es "sin etiquetas"), y ningun texto libre cruza
    la frontera hacia el solver, los vetos van tipados
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# diagnostico (Agente 1)
class TipoPrecursor(str, Enum):
    """taxonomia calibrada empiricamente en los notebooks 1-4 (umbrales en xai_tools)."""

    CANAL_UNICO = "canal_unico"  # CF top-1 explica ~>80% del score (ev66 93%)
    GRUPO_COHERENTE = "grupo_coherente"  # CF acumulado escalonado (ev11 64, 86, 92%)
    DIFUSO = "difuso"  # CF top-5 < ~50% (B ev53 24, 41%)
    RELACIONAL = "relacional"  # IG, el contexto domina y el canal propio ~<10% (C ev11)
    MIXTO = "mixto"  # IG, canal propio 20-50% (B 29-34%)
    SIN_SENAL = "sin_senal"  # margen S/h < 1 o sin alarma


class ChannelZ(BaseModel):
    canal: str
    z: float


class AlarmEvidence(BaseModel):
    """expediente objetivo de una alarma, todo computado por tools XAI y nada del LLM.
    label-free por diseno, sin etiquetas CARE ni nada derivado, que es la condicion de validez
    de la eval de triaje (F4)."""

    farm: str = Field(pattern="^[BC]$")
    event_id: int
    asset_id: int
    detector: str = Field(description="score que disparo la alarma, p.ej. 'fusion_ae_crgn_phys95maxrate' (C) | 'ae_ident_s42' (B)")
    t_alarma: datetime | None = None
    margen_sh: float = Field(description="max CUSUM / umbral h en el evento")
    persistencia_frac: float | None = Field(default=None, ge=0.0, le=1.0,
                                            description="fraccion de ventanas TRAS la alarma con S>h "
                                                        "(sostenida ~ precursor real; transitoria ~ falsa alarma)")
    crecimiento_z: float | None = Field(default=None,
                                        description="max z_slope (tasa del |error| de pronostico, "
                                                    "leakage-free) en la ventana de alarma: alto = trayectoria "
                                                    "DIVERGIENDO (precursor acelera); ~0 = plana (falsa alarma)")
    lead_h: float | None = Field(default=None,
                                 description="antelacion (h) vs FIN DE LA VENTANA DE PREDICCION del evento "
                                             "(estructura del dataset, conocida sin etiqueta)")
    canal_disparador: str | None = None
    top_canales: list[ChannelZ] = Field(default_factory=list)
    cf_acumulado_pct: list[float] = Field(default_factory=list,
                                          description="caida % del score al neutralizar top-k, k=1..K")
    ig_contexto_pct: float | None = Field(default=None,
                                          description="% de la atribucion IG total en canales de contexto (!= propio)")
    ig_propio_pct: float | None = Field(default=None,
                                        description="% de la atribucion IG total en el canal propio (relacional <~10; mixto 20-50)")
    miembro_dominante: str | None = Field(default=None, description="en ensembles: ae/crgn/phys")
    notas: list[str] = Field(default_factory=list)


class TriageAction(str, Enum):
    ESCALAR = "escalar"
    VIGILAR = "vigilar"
    DESCARTAR = "descartar"


class TriageDecision(BaseModel):
    action: TriageAction
    razones: list[str] = Field(min_length=1)


class AlarmReport(BaseModel):
    """salida del agente 1, interpretacion y triaje sobre la evidencia ya validada."""

    evidencia: AlarmEvidence
    tipo_precursor: TipoPrecursor
    confianza: float = Field(ge=0.0, le=1.0)
    hipotesis: str = Field(description="diagnostico en 1-3 frases, citando SOLO numeros de la evidencia")
    triage: TriageDecision
    urgencia_dias: int | None = Field(default=None, ge=0,
                                      description="tiempo estimado hasta el FALLO si no se actua (dias); "
                                                  "heuristica del lead del detector (nuestra alarma avisa "
                                                  "con antelacion), assumption, no un oraculo. El derating "
                                                  "alarga esta ventana en la practica")
    trace_id: str | None = None

    @model_validator(mode="after")
    def _escalar_requiere_urgencia(self) -> "AlarmReport":
        if self.triage.action is TriageAction.ESCALAR and self.urgencia_dias is None:
            raise ValueError("triage 'escalar' exige urgencia_dias (conecta con la ventana de la tarea)")
        return self


class EvalContext(BaseModel):
    """verdad-terreno CARE de un evento, solo para evaluacion y notebooks (CARE post-filtro y
    casos de estudio). fuera de AlarmEvidence a proposito, jamas entra en un prompt."""

    farm: str = Field(pattern="^[BC]$")
    event_id: int
    es_evento_normal: bool
    lead_real_h: float | None = Field(default=None,
                                      description="antelacion vs etiqueta real, solo eventos anomalos")


# entorno (KB ambiental)
class EnvPeriod(BaseModel):
    """periodo critico de una especie, cada entrada lleva su fuente. con mes_fin < mes_inicio
    el periodo cruza el ano (nov-feb), _month_in resuelve la pertenencia."""

    especie: str
    nombre_cientifico: str | None = None
    actividad: str = Field(description="reproduccion/alimentacion/nidificacion/migracion")
    mes_inicio: int = Field(ge=1, le=12)
    mes_fin: int = Field(ge=1, le=12)
    sensibilidad: float = Field(ge=0.0, le=1.0, description="peso del periodo en el objetivo ambiental")
    canal_impacto: Literal["ruido_operacional", "trafico_embarcaciones", "ambos"]
    fuente: str
    assumption: bool = Field(default=False, description="True si el parametro no es citable directo")


class NoiseRule(BaseModel):
    nombre: str
    umbral_db: float
    metrica: str = Field(description="pico re 1uPa | SEL re 1uPa2s")
    distancia_m: float | None = None
    contexto: str
    fuente: str


# scheduling (Agente 2 y solver)
class MaintenanceTask(BaseModel):
    """tarea derivada de un AlarmReport escalado, o preventiva."""

    task_id: str
    farm: str
    asset_id: int
    origen_event_id: int | None = None
    subsistema: str = "desconocido"
    duracion_dias: int = Field(ge=1)
    ventana_inicio_dia: int = Field(ge=0, description="dia mas temprano posible (desde t0)")
    ventana_fin_dia: int = Field(description="ULTIMO dia de INICIO permitido (cota superior de la "
                                             "variable de decision; no es el fin de ejecucion — el "
                                             "riesgo sigue creciendo si se apura)")
    coste_intervencion: float = Field(description="coste CIERTO de la intervencion planificada "
                                                  "(movilizacion CTV/jack-up + mano de obra + repuestos)")
    riesgo_fallo_base: float = Field(ge=0, le=1, description="prob. fallo en horizonte si no se actua")
    riesgo_crecimiento_dia: float = Field(default=0.0, ge=0,
                                          description="pendiente que FIJA el deadline de rotura: el riesgo "
                                                      "llega al techo (1.0) en (techo-base)/crec dias. La "
                                                      "curva real es CONVEXA hasta ahi (problem.py RIESGO_EXP)")
    coste_generacion_dia: float = Field(default=0.0, ge=0,
                                        description="lucro cesante por dia de PARADA de la turbina "
                                                    "degradada (politica: al escalar se para/derata); "
                                                    "coste CIERTO que crece al diferir (refinamiento F5)")
    coste_correctivo: float = Field(description="coste del correctivo CATASTROFICO si falla antes de "
                                                "reparar (probabilistico: se pondera por p_fallo). Ya "
                                                "incluye la prima de reparacion de emergencia y el dano "
                                                "secundario: es varias veces la intervencion planificada")
    requiere_jackup: bool = False
    # la degradada no se para del todo mientras espera, opera derateada y eso reduce carga y
    # dano (practica real), la reparacion planificada evita el fallo y deja riesgo residual
    derate_factor: float = Field(default=0.30, ge=0.0, le=1.0,
                                 description="fraccion de generacion PERDIDA operando derateada hasta la "
                                             "reparacion planificada (no es parada total: se genera el resto)")
    p_fallo_max: float = Field(default=1.0, ge=0.0, le=1.0,
                               description="prob. de fallo en la fecha de rotura EXTENDIDA por el derating "
                                           "(=1.0: la pieza rompe si no se repara a tiempo). El derating NO "
                                           "baja este techo, solo aleja CUANDO se alcanza. La hibrida acota el "
                                           "riesgo antes (para la turbina); el derate puro puede llegar a 1.0")
    factor_ventana_derating: float = Field(default=2.0, ge=1.0,
                                           description="el derating FRENA la degradacion -> el fallo se aleja: "
                                                       "el riesgo llega a 1.0 en ~factor x urgencia (no "
                                                       "en la urgencia cruda). Es lo que da margen para planificar. "
                                                       "Anclado a ISO 281: reducir la carga ~20% (derate_factor 0.20) "
                                                       "~= 2x vida del rodamiento -> factor ~= 2 (antes 4.0 a ojo)")
    espera_emergencia_dias: int = Field(default=14, ge=0,
                                        description="dias extra de parada de una reparacion de EMERGENCIA "
                                                    "(no planificada): movilizacion de barco + ventana meteo "
                                                    "(jack-up ~45, CTV ~14). Solo lo usa el reactivo (savings)")
    # un buque grande no se contrata para manana a precio planificado, intervenir antes del
    # lead paga prima y tarde sube derating, riesgo e impacto, hay un punto dulce en medio
    lead_planificacion_dias: int = Field(default=7, ge=0,
                                         description="dias minimos para contratar el recurso a precio "
                                                     "planificado (jack-up ~30, CTV ~7)")
    prima_urgencia_max: float = Field(default=2.0, ge=1.0,
                                      description="multiplicador MAXIMO del coste de intervencion si se "
                                                  "hace en el dia 0 (expedita); baja a 1.0 al alcanzar el lead")
    # impacto acustico de la intervencion segun el buque (f3), un jack-up (izado pesado, patas)
    # es la operacion mas ruidosa e intrusiva y un CTV perturba mucho menos
    factor_ruido_buque: float = Field(default=0.3, ge=0.0, le=1.0,
                                      description="peso del ruido de la intervencion segun el buque "
                                                  "(jack-up ~1.0, CTV ~0.3) en f3")
    factor_trafico_buque: float = Field(default=0.6, ge=0.0, le=1.0,
                                        description="peso del trafico de la intervencion segun el buque en f3")

    @model_validator(mode="after")
    def _ventana_coherente(self) -> "MaintenanceTask":
        if self.ventana_fin_dia < self.ventana_inicio_dia:
            raise ValueError(f"ventana_fin_dia ({self.ventana_fin_dia}) < ventana_inicio_dia "
                             f"({self.ventana_inicio_dia}) en {self.task_id}")
        return self


class Recursos(BaseModel):
    """recursos de la campana. los defaults son assumption de trabajo y pasan a una tabla
    citable con fuente y assumption por parametro."""

    n_ctv: int = 1
    coste_dia_ctv: float = 5000.0
    max_tareas_por_dia: int = 2


class VetoDuro(BaseModel):
    """veto duro tipado, nada de texto libre hacia el solver. problem.py descarta como dia de
    inicio los dias cuyo mes cae en [mes_inicio, mes_fin] para las tareas del alcance."""

    # con mes_fin < mes_inicio el veto cruza el ano, la misma convencion que EnvPeriod
    alcance: Literal["jackup", "todas"]
    mes_inicio: int = Field(ge=1, le=12)
    mes_fin: int = Field(ge=1, le=12)
    motivo: str = Field(description="p.ej. 'reproduccion marsopa jun-ago (KB)'")


class PlanRequest(BaseModel):
    """instancia completa del problema que consume el solver, la construye el agente 2."""

    t0: date
    horizonte_dias: int = Field(ge=7)
    tasks: list[MaintenanceTask] = Field(min_length=1)
    recursos: Recursos = Recursos()
    n_escenarios_meteo: int = 50
    weather_provider: Literal["synthetic", "era5"] = "synthetic"
    seed: int = 42
    periodos_ambientales: list[EnvPeriod] = Field(default_factory=list)
    vetos_duros: list[VetoDuro] = Field(default_factory=list)
    politica_espera: Literal["derate", "stop", "hibrida", "hibrida_opt"] = Field(default="derate",
        description="mientras espera la reparacion: 'derate' (opera a potencia reducida, pierde "
                    "derate_factor, riesgo residual + ruido continuo) | 'stop' (parada total: pierde el "
                    "100% de produccion pero riesgo ~0 y sin ruido -> esperar la mejor ventana sin riesgo) | "
                    "'hibrida' (derate hasta que el riesgo alcanza un umbral FIJO de seguridad y ENTONCES "
                    "para: no se opera mas alla de la ventana de fallo; riesgo acotado + algo de produccion "
                    "antes de parar) | 'hibrida_opt' (como hibrida pero el dia de PARADA es una variable de "
                    "decision mas, ACOTADA por el umbral de seguridad: el optimizador puede parar ANTES para "
                    "bajar riesgo/ruido a costa de produccion, nunca despues -> barandilla dura + parada "
                    "optimizada)")


class ScheduledTask(BaseModel):
    task_id: str
    dia_inicio: int = Field(ge=0)
    dia_parada: int | None = Field(default=None, ge=0,
                                   description="solo politica 'hibrida_opt': dia (desde t0) en que se PARA "
                                               "la turbina degradada mientras espera la reparacion (variable "
                                               "de decision, acotada por el umbral de seguridad). None en el "
                                               "resto de politicas (la parada la fija su regla, no se optimiza)")


class Objetivos(BaseModel):
    coste_esperado: float
    riesgo_esperado: float
    impacto_ambiental: float


class CandidatePlan(BaseModel):
    schedule: list[ScheduledTask]
    objetivos: Objetivos


class ValidacionExacta(BaseModel):
    """resultado del exact_check en el sub-caso determinista, el gate maquina-verificable de
    F3."""

    optimo_alcanzado: bool
    gap: float = Field(ge=0, description="distancia relativa del frente al optimo exacto (0 = lo toca)")
    detalle: str | None = None


class ParetoFront(BaseModel):
    request_resumen: str
    soluciones: list[CandidatePlan]
    knee_idx: int | None = None
    baseline_greedy: CandidatePlan | None = None
    validacion_exacta: ValidacionExacta | None = None

    @model_validator(mode="after")
    def _knee_en_rango(self) -> "ParetoFront":
        if self.knee_idx is not None and not 0 <= self.knee_idx < len(self.soluciones):
            raise ValueError(f"knee_idx {self.knee_idx} fuera de rango (0..{len(self.soluciones) - 1})")
        return self


class MaintenancePlan(BaseModel):
    """salida del agente 2, la solucion elegida del frente y su explicacion trazable."""

    elegido: CandidatePlan
    narrativa: str = Field(description="explicacion del trade-off citando SOLO numeros del frente")
    alternativas: list[str] = Field(default_factory=list,
                                    description="resumen 1-linea de 2-3 soluciones alternativas del frente")
    reports_origen: list[int] = Field(default_factory=list, description="event_ids diagnosticados")
    trace_id: str | None = None
