"""
trabajos pesados (torch y NSGA-II) fuera del event-loop, en un unico worker, asi la GPU y el
solver se serializan y nunca compiten dos solves por la tarjeta. los endpoints async hacen
`await run(fn, ...)` y no bloquean el servidor mientras el hilo trabaja.

sencillo a proposito, por decision de diseno, el registro de jobs vive en proceso, sin Celery
ni Redis.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

# un solo hilo, es lo que serializa la GPU y el solver
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cockpit")


async def run(fn: Callable[..., Any], *args: Any) -> Any:
    """ejecuta fn(*args) en el worker unico y devuelve el resultado, sin bloquear el
    event-loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, fn, *args)
