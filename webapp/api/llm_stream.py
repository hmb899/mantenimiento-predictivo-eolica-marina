"""
terminal LLM acotado al dominio (qwen3 local via Ollama), con streaming token a token para el
efecto de consola CRT. vive aqui, en webapp/, y no en src/, porque el OllamaClient de src/ va
sin stream.

por diseno el asistente responde solo sobre esta alarma, esta turbina y su mantenimiento,
citando unicamente los numeros del expediente que se le pasa, y a lo ajeno contesta con la
frase fija OUT_OF_SCOPE. la robustez va en dos capas, primero un filtro de relevancia barato
(heuristica de palabras y mini-llamada si/no) que corta las preguntas obviamente ajenas sin
generar, y despues el system prompt, que fuerza el rechazo si algo se cuela.
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

from src.agents.llm import DEFAULT_MODEL, OLLAMA_URL

from . import context as ctx

OUT_OF_SCOPE = "Solo puedo ayudarte con esta alarma y su mantenimiento."

_SYSTEM = (
    "Eres el asistente de una consola de mantenimiento predictivo de aerogeneradores offshore. "
    "SOLO respondes sobre ESTA alarma, esta turbina y su mantenimiento, usando EXCLUSIVAMENTE los "
    "datos del expediente que te doy. Reglas que NO puedes romper: (1) cita solo numeros presentes "
    "en el expediente, jamas inventes valores; (2) el sensor es un INDICADOR, nunca digas "
    "'revisar/cambiar el sensor': lo que se degrada es el COMPONENTE que vigila; (3) responde breve "
    "(2-4 frases), tecnico y claro, sin markdown; (4) si te preguntan algo ajeno a esta alarma, esta "
    f"turbina o su mantenimiento, responde EXACTAMENTE: \"{OUT_OF_SCOPE}\""
)

# palabras del dominio, si aparece alguna la pregunta es del ambito y nos ahorramos
# el mini-clasificador
_DOMINIO = (
    "turbina aerogenerador alarma sensor cusum margen coste riesgo impacto ambiental fauna marsopa "
    "foca reparar reparacion derat parar paron plan planific fallo romper ventana ctv viento oleaje "
    "hs meteo era5 mantenimiento jack subsistema multiplicadora rodamiento generador convertidor "
    "precursor relacional counterfactual evento knee pareto urgencia catastrofico correctivo "
    "predictivo diagnostic anomalia esperar diferir intervenir buque potencia derateo derateada"
).split()


def _client_cfg() -> tuple[str, str]:
    c = ctx.ollama_client()
    return c.url, c.model


def in_scope(question: str) -> bool:
    """filtro de relevancia, primero la heuristica barata y si es ambigua una mini-llamada
    si/no. ante cualquier fallo deja pasar, que el system prompt es el backstop de lo ajeno."""
    q = question.lower()
    if any(w in q for w in _DOMINIO):
        return True
    try:
        url, model = _client_cfg()
        sys = ("Clasifica si la PREGUNTA trata sobre el mantenimiento de un aerogenerador, su alarma, "
               "diagnostico o reparacion, el coste/riesgo/impacto de intervenir, o la meteorologia de "
               "acceso a la turbina. Responde SOLO con 'si' o 'no'.")
        r = requests.post(f"{url}/api/chat", timeout=30, json={
            "model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": sys},
                         {"role": "user", "content": question}]})
        r.raise_for_status()
        ans = r.json()["message"]["content"].strip().lower()
        return not ans.startswith("no")
    except Exception:
        return True


def build_context(farm: str, event: int, seed: int, t0_iso: str, con: bool, weather: str,
                  nsc: int, horizonte: int, mes: int) -> str:
    """expediente compacto del caso, los numeros ya computados que anclan al asistente."""
    rep = ctx.diagnose(farm, event, seed, False)
    ev = rep.evidencia
    lineas = [
        f"TURBINA: {farm}-{ev.asset_id} (Farm {farm}). EVENTO/ALARMA: ev{event}. Detector: {ev.detector}.",
        f"Margen S/h: {ev.margen_sh} (>=1 = alarma). Canal disparador (indicador): {ev.canal_disparador}.",
        f"Tipo de precursor: {rep.tipo_precursor.value}. Triaje: {rep.triage.action.value} "
        f"({'; '.join(rep.triage.razones)}).",
        f"IG propio: {ev.ig_propio_pct}% / IG contexto: {ev.ig_contexto_pct}% "
        "(<10% propio = relacional: el sensor no se desvia solo, el resto de la turbina deja de predecirlo).",
        f"Counterfactual acumulado top-k (% caida del score): {ev.cf_acumulado_pct}.",
        f"Persistencia tras la alarma: {ev.persistencia_frac}. Lead: {ev.lead_h} h.",
        # chr(10) es el salto de linea, la hipotesis viene en varias lineas y aqui va en una
        f"Diagnostico redactado: {rep.hipotesis.replace(chr(10), ' ')}",
    ]
    po = ctx.plan_options(farm, event, t0_iso, con, weather, nsc, seed, horizonte, mes)
    if po is not None:
        lineas.append(f"Subsistema representativo: {po['subsistema']} "
                      f"({'jack-up' if po['requiere_jackup'] else 'CTV'}). "
                      f"Urgencia (tiempo a fallo sin actuar): {po['urgencia_dias']} d; "
                      f"con derating se aleja a ~{po['urgencia_derating_dias']} d.")
        for o in po["opciones"]:
            c_ = f"{o['coste']:,.0f}" if o.get("coste") is not None else "-"
            lineas.append(f"Opcion '{o['titulo']}': coste {c_}, riesgo {o['riesgo']} "
                          f"({o['riesgo_cual']}), impacto {o.get('impacto')} ({o['impacto_cual']}), "
                          f"cuando {o['cuando']}.")
        lineas.append("Nota: canales anonimizados -> subsistema y costes son REPRESENTATIVOS y "
                      "RELATIVOS. El 'optimo' es el recomendado (equilibra coste/riesgo/impacto).")
    wx = ctx.weather_month(mes)
    if wx is not None:
        lineas.append(f"Meteo de acceso (ERA5, mes {wx['mes']}): viento {wx['viento']} m/s, "
                      f"Hs {wx['hs']} m, {wx['pct_accesible']}% dias accesibles CTV, "
                      f"ventana CTV {'SI' if wx['ventana_ctv'] else 'NO'}.")
    return "\n".join(lineas)


def _raw_stream(messages: list[dict]) -> Iterator[str]:
    url, model = _client_cfg()
    payload = {"model": model, "messages": messages, "stream": True, "think": False,
               "options": {"temperature": 0.2}}
    # ollama devuelve un json por linea, se ignora lo que no parsee y se corta en done
    with requests.post(f"{url}/api/chat", json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


def _partial_tail(s: str, tag: str) -> int:
    """longitud del sufijo de s que es prefijo de tag, para no partir una etiqueta."""
    # del sufijo mas largo posible hacia abajo, el primero que encaja es el que se retiene
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if tag.startswith(s[-k:]):
            return k
    return 0


def _strip_think(gen: Iterator[str]) -> Iterator[str]:
    """elimina los bloques <think>...</think> del stream, por si el modelo los emite pese a
    think=false, y tolera las etiquetas partidas entre chunks."""
    # automata de dos estados sobre un buffer, hiding dice si estamos dentro de un <think>
    buf, hiding = "", False
    for chunk in gen:
        buf += chunk
        while True:
            if not hiding:
                # fuera del bloque, se emite todo menos la cola que puede ser media etiqueta
                i = buf.find("<think>")
                if i == -1:
                    keep = _partial_tail(buf, "<think>")
                    emit, buf = buf[:len(buf) - keep], buf[len(buf) - keep:]
                    if emit:
                        yield emit
                    break
                if buf[:i]:
                    yield buf[:i]
                buf, hiding = buf[i + 7:], True   # 7 = len("<think>")
            else:
                # dentro del bloque no se emite nada, solo se busca el cierre
                j = buf.find("</think>")
                if j == -1:
                    keep = _partial_tail(buf, "</think>")
                    buf = buf[len(buf) - keep:]
                    break
                buf, hiding = buf[j + 8:], False  # 8 = len("</think>")
    if not hiding and buf:
        yield buf


def stream_answer(contexto: str, question: str) -> Iterator[str]:
    """genera la respuesta del asistente acotada al expediente, token a token y sin <think>."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "EXPEDIENTE DEL CASO:\n" + contexto +
         "\n\nPREGUNTA DEL OPERADOR: " + question},
    ]
    yield from _strip_think(_raw_stream(messages))
