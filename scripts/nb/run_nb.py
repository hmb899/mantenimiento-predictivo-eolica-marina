"""Ejecuta un notebook con nbclient (kernel del venv, cwd = carpeta del notebook) y guarda
outputs en el propio .ipynb. Uso: python scripts/nb/run_nb.py notebooks/01_xxx.ipynb [timeout_s]"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    nb_path = ROOT / sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
    nb = nbf.read(str(nb_path), as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(nb_path.parent)}})
    t0 = time.time()
    try:
        client.execute()
    finally:
        nbf.write(nb, str(nb_path))
        n_err = sum(1 for c in nb.cells if c.cell_type == "code"
                    for o in c.get("outputs", []) if o.get("output_type") == "error")
        print(f"ejecutado en {time.time()-t0:.0f}s | celdas {len(nb.cells)} | errores {n_err} -> {nb_path}")
        if n_err:
            for i, c in enumerate(nb.cells):
                if c.cell_type != "code":
                    continue
                for o in c.get("outputs", []):
                    if o.get("output_type") == "error":
                        print(f"--- ERROR en celda {i}: {o.get('ename')}: {o.get('evalue')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
