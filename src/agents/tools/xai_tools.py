"""
tools de diagnostico del Agente 1

compilan el expediente objetivo y label-free de una alarma llamando a src/xai. la evidencia
es reproducible, con la misma receta que los notebooks 1/3 y verificada por
scripts/verify_xai_loaders.py, y no usa etiquetas. el punto de operacion sale de
official_cusum, percentil-99 del train, sin CARE-oracle y por tanto sin fuga, y el baseline
del IG son las ventanas tempranas del evento, pre-alarma, no las normales por etiqueta.

recetas, identicas a las del NB1/NB3
    alarma y margen      official_cusum(train) da (mu, k, h_base), luego el CUSUM del evento,
                         first_alarm y S.max()/h_base
    canal disparador     trajdev_channel_z(err)["z"] en la ventana de alarma, argmax sobre los
                         canales val
    tipo de precursor    trajdev_counterfactual acumulado top-k, la caida en % del score, o sea
                         cf_acumulado_pct
    local vs relacional  trajdev_integrated_gradients, el % del canal propio en el contexto

el detector interpretable por construccion es TrajDev, donde el z por canal es atribucion
exacta, en las dos granjas. el campo `detector` registra cual score dispara.
"""
from __future__ import annotations

import numpy as np

from src.agents.contracts import AlarmEvidence, ChannelZ, TipoPrecursor

_CTX: dict[tuple, dict] = {}  # cache por (granja, detector), se construye una vez

# detector por defecto = campeon dep_zmax_val (own3_val en C, ownb3_zmax en B). fija el punto
# de operacion, margen, alarma y persistencia, pero la evidencia (z por canal, CF, IG) sale del
# bundle y no depende de la agregacion, asi que own3_all, mas temprano y ruidoso, la reusa
_DET_KEY = {"C": "own3_val", "B": "ownb3_zmax"}


def _context(farm: str, det_key: str | None = None, seed: int = 42) -> dict:
    """construye, una vez por (granja, detector, seed), el contexto pesado: bundle TrajDev,
    meta, eventos en el espacio del modelo y el punto de operacion label-free (official_cusum
    del train). seed != 42 usa el bundle y los scores re-entrenados de esa semilla, para el
    multiseed del triaje."""

    dk = det_key or _DET_KEY[farm]
    ck = (farm, dk, seed)
    if ck in _CTX:
        return _CTX[ck]
    from src.xai import (load_meta, load_scores, load_trajdev, load_trajdev_events)
    from src.xai.cusum import cusum_series, first_alarm, official_cusum

    bundle = load_trajdev(farm, seed)
    meta = load_meta(farm)
    events = {e["event_id"]: e for e in load_trajdev_events(farm)}
    train_dump, ev_dump = load_scores(farm, dk, seed)
    # el umbral se calibra solo con el train, que es lo que hace label-free al expediente
    mu, k, h_base = official_cusum(train_dump)
    ev_S = {e: cusum_series(s, mu, k) for e, s in ev_dump.items()}
    val_mask = bundle["masks"].get("val")
    _CTX[ck] = {"bundle": bundle, "meta": meta, "events": events, "mu": mu, "k": k, "det": dk,
                "h": h_base, "ev_S": ev_S, "val_mask": val_mask, "first_alarm": first_alarm}
    return _CTX[ck]


def _cf_acumulado(bundle, wins, cand: list[int]) -> list[float]:
    """contrafactual acumulado top-k, la caida en % del score al sustituir los k canales top
    por su propio pronostico (test de suficiencia de grupo, NB1 §6). cand ya viene ordenado por
    z descendente."""

    from src.xai.attrib import trajdev_counterfactual
    from src.xai.loaders import trajdev_channel_z, trajdev_forecast_errors, trajdev_score

    def score(w):
        err, _ = trajdev_forecast_errors(bundle, w)
        return float(trajdev_score(bundle, trajdev_channel_z(bundle, err)["z"], "dep_zmax_val").mean())

    base = score(wins)
    if base <= 0:
        return []
    out = []
    # acumulado y no uno a uno, en cada vuelta se ocluyen los kk primeros a la vez, tope 6
    for kk in range(1, min(len(cand), 6) + 1):
        cf = trajdev_counterfactual(bundle, wins, cand[:kk])
        out.append(round(100.0 * (1.0 - score(cf) / base), 1))
    return out


def _ig_local_relacional(bundle, ev_data, i_ref: int, c_top: int) -> tuple[float | None, float | None]:
    """% de la atribucion IG del canal propio dentro del contexto, relacional si baja de ~10%,
    y % del contexto sobre el total (NB1 §5.1). baseline label-free = mediana de las 400
    primeras ventanas."""

    from src.xai.attrib import trajdev_integrated_gradients

    ctx = bundle["arch"]["context"]
    # las 400 primeras son las mas tempranas del evento, aun sin fallo, y no llevan etiqueta
    base = np.median(ev_data[:400], axis=0)
    att = np.abs(trajdev_integrated_gradients(bundle, ev_data[i_ref:i_ref + 1], c_top,
                                              baseline_window=base, steps=50))[0]
    # solo el tramo de contexto, [:ctx], habla de que sensores predicen; el resto es el
    # objetivo
    ctx_sum = att[:ctx, :].sum()
    if ctx_sum <= 0:
        return None, None
    propio = 100.0 * att[:ctx, c_top].sum() / ctx_sum
    contexto = 100.0 * ctx_sum / att.sum()
    return round(float(contexto), 1), round(float(propio), 1)


def get_alarm_evidence(farm: str, event_id: int, det_key: str | None = None,
                       with_ig: bool = True, seed: int = 42) -> AlarmEvidence:
    """expediente completo y label-free de la alarma de un evento. det_key es la variante de
    score (None = campeon dep_zmax_val, por ejemplo 'own3_all' = dep_zmax_all, mas temprano) y
    seed elige el bundle y los scores de esa semilla, para el multiseed del triaje."""

    from src.xai.loaders import trajdev_channel_z, trajdev_forecast_errors

    c = _context(farm, det_key, seed)
    bundle, ev, m = c["bundle"], c["events"][event_id], c["meta"][event_id]
    S, h = c["ev_S"][event_id], c["h"]
    idx = c["first_alarm"](S, h)
    margen = float(S.max() / h)
    # sin alarma se usa el maximo del CUSUM como referencia, para poder describir el evento
    # igual
    i_ref = idx if idx is not None else int(S.argmax())
    # persistencia, fraccion de ventanas tras la alarma que siguen sobre el umbral (label-free)
    persistencia = float((S[idx:] > h).mean()) if idx is not None else None

    err, _ = trajdev_forecast_errors(bundle, ev["data"])
    zfam = trajdev_channel_z(bundle, err)
    z, z_slope = zfam["z"], zfam["z_slope"]
    vm = c["val_mask"]
    val_idx = np.flatnonzero(vm) if vm is not None else np.arange(z.shape[1])

    # canal disparador y top-5 por z medio en la ventana de alarma (+-3)
    sl = slice(max(0, i_ref - 3), min(len(z), i_ref + 4))
    zbar = z[sl].mean(axis=0)
    # se rankea solo sobre val, pero se indexa zbar con el indice global, no con el del
    # subconjunto
    zbar_val = zbar[val_idx]
    order = val_idx[np.argsort(zbar_val)[::-1]]
    c_top = int(order[0])
    top = [ChannelZ(canal=bundle["ch_names"][ci], z=round(float(zbar[ci]), 2)) for ci in order[:5]]
    # crecimiento (palanca 2, label-free), tasa del |error| divergiendo en la alarma, max de
    # z_slope sobre val, alto = precursor acelera, ~0 = excursion plana que revierte
    crecimiento = float(z_slope[sl].mean(axis=0)[val_idx].max()) if margen >= 1.0 else None

    # tipo de precursor, contrafactual acumulado sobre 24 ventanas desde la alarma
    wins = ev["data"][i_ref:i_ref + 24]
    cand = [int(ci) for ci in order[:6]]
    # margen < 1 significa que no hubo alarma, y entonces no hay nada que caracterizar
    cf_acc = _cf_acumulado(bundle, wins, cand) if margen >= 1.0 else []

    ig_ctx, ig_prop = (None, None)
    if with_ig and margen >= 1.0:
        ig_ctx, ig_prop = _ig_local_relacional(bundle, ev["data"], i_ref, c_top)

    # lead = horas entre la alarma y el final de la ventana del evento, o sea la antelacion
    lead_h = None
    if idx is not None:
        lead_h = round((m["window_timestamps"][-1] - m["window_timestamps"][idx]).total_seconds()
                       / 3600.0, 1)

    return AlarmEvidence(
        farm=farm, event_id=event_id, asset_id=int(m["asset_id"]), detector=f"trajdev_{c['det']}",
        t_alarma=(m["window_timestamps"][idx].to_pydatetime() if idx is not None else None),
        margen_sh=round(margen, 3),
        persistencia_frac=(round(persistencia, 3) if persistencia is not None else None),
        crecimiento_z=(round(crecimiento, 2) if crecimiento is not None else None),
        lead_h=lead_h,
        canal_disparador=(bundle["ch_names"][c_top] if margen >= 1.0 else None),
        top_canales=top, cf_acumulado_pct=cf_acc, ig_contexto_pct=ig_ctx, ig_propio_pct=ig_prop,
        notas=[f"punto de operacion label-free h_base={h:.3f}",
               "sin alarma al h_base" if idx is None else f"alarma en ventana {idx}"])


def clasificar_precursor(ev: AlarmEvidence) -> tuple[TipoPrecursor, float]:
    """clasificacion determinista, el fallback del agente, con los umbrales calibrados en
    NB1-NB4. devuelve (tipo, confianza heuristica) sobre dos ejes, concentracion (CF acumulado)
    y localidad (IG). el orden de prioridad es sin_senal, relacional o mixto si hay IG,
    canal_unico, difuso y por ultimo grupo."""

    if ev.margen_sh < 1.0 or ev.canal_disparador is None:
        return TipoPrecursor.SIN_SENAL, 0.9
    cf = ev.cf_acumulado_pct
    # top1 es la caida ocluyendo solo el mejor canal, top5 la de los cinco primeros acumulados
    top1 = cf[0] if cf else 0.0
    top5 = cf[min(4, len(cf) - 1)] if cf else 0.0

    # la localidad manda sobre la concentracion, un precursor relacional se reconoce por el IG
    if ev.ig_propio_pct is not None:
        if ev.ig_propio_pct < 10.0:
            return TipoPrecursor.RELACIONAL, 0.8
        if 20.0 <= ev.ig_propio_pct <= 50.0:
            return TipoPrecursor.MIXTO, 0.7
    if top1 >= 80.0:
        return TipoPrecursor.CANAL_UNICO, 0.85
    if cf and top5 < 50.0:
        return TipoPrecursor.DIFUSO, 0.7
    return TipoPrecursor.GRUPO_COHERENTE, 0.75
