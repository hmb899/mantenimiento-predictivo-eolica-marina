"""
cliente LLM backend-agnostico

regla de la fase, toda salida de LLM se valida contra un contrato pydantic antes de entrar al
pipeline, y si tras los reintentos no valida el llamador usa su fallback determinista, de modo
que el LLM nunca es un punto unico de fallo

OllamaClient
    backend por defecto, Ollama local (en dev qwen3:latest, ya instalado, opcion qwen2.5:14b)
"""
from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:latest"


class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kw) -> str: ...


    def structured(self, messages: list[dict], schema: type[T], retries: int = 2, **kw) -> T: ...


def _extract_json(text: str) -> str:
    """extrae el primer bloque JSON plausible (modelos locales a veces envuelven en
    ```json)."""

    # primero el bloque vallado, que es lo que devuelve un modelo que se pone conversacional
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # si no lo hay, se recorta entre la primera llave y la ultima
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


class OllamaClient:
    """cliente minimo contra la API local de Ollama (/api/chat), sin dependencias extra."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL, timeout: int = 180,
                 temperature: float = 0.2) -> None:
        self.model, self.url, self.timeout, self.temperature = model, url, timeout, temperature


    def ping(self) -> list[str]:
        # lista los modelos instalados, o sea que vale como comprobacion de vida del servidor
        r = requests.get(f"{self.url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


    def chat(self, messages: list[dict], json_mode: bool = False, **kw) -> str:
        # kw pisa al modelo y la temperatura por llamada, sin tocar los del constructor
        payload = {"model": kw.get("model", self.model), "messages": messages, "stream": False,
                   "options": {"temperature": kw.get("temperature", self.temperature)}}
        # json_mode le pide a Ollama que fuerce salida JSON, no que cumpla ningun schema
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


    def structured(self, messages: list[dict], schema: type[T], retries: int = 2, **kw) -> T:
        """chat con contrato, pide JSON conforme al schema y valida, reintentando con el error
        como feedback. lanza si agota reintentos y el llamador hace fallback."""

        # el schema entero se anexa al ultimo mensaje del usuario, no al system
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        msgs = list(messages)
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] +
                    "\n\nRESPONDE UNICAMENTE con un objeto JSON valido conforme a este JSON-Schema "
                    "(sin texto adicional, sin markdown):\n" + schema_json}
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(msgs, json_mode=True, **kw)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_err = exc
                # el error vuelve al modelo como feedback, y cada reintento arrastra los
                # anteriores
                msgs = msgs + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": f"JSON invalido ({exc}). Corrige y "
                                "devuelve SOLO el objeto JSON valido."}]
        raise ValueError(f"salida LLM no validable tras {retries + 1} intentos: {last_err}")
