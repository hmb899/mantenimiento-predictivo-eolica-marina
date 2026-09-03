"""smoke y gate del scheduler (F3), sobre un caso realista y el provider sintetico.

el caso son tareas con ventanas y riesgos, una con jack-up, veto de jack-up jun-ago y t0 en
junio; sobre el corre solve_plan (NSGA-II + greedy + validacion exacta) con el provider
sintetico, rapido y sin depender de la descarga ERA5. se invoca con python
scripts/scheduler_smoke.py, y el gate que verifica es:

  [1] el frente no esta vacio y el knee es valido
  [2] alguna solucion del frente domina (debilmente) al baseline greedy
  [3] el frente ofrece un trade-off real (min-coste != min-impacto)
  [4] la validacion exacta alcanza el optimo (gap ~ 0) en el sub-caso
"""
from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.agents.contracts import MaintenanceTask, PlanRequest, VetoDuro
from src.agents.tools.env_tools import load_kb
from src.agents.tools.solver_tools import solve_plan

FAILS = []


def check(name, ok, detail=""):
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(name)


def _obj(cp):
    o = cp.objetivos
    return np.array([o.coste_esperado, o.riesgo_esperado, o.impacto_ambiental])


def main() -> None:
    periodos = load_kb()[0]

    def task(tid, **kw):
        d = dict(task_id=tid, farm="C", asset_id=13, duracion_dias=2, ventana_inicio_dia=0,
                 ventana_fin_dia=140, coste_intervencion=80_000, riesgo_fallo_base=0.10,
                 riesgo_crecimiento_dia=0.004, coste_generacion_dia=9_600, coste_correctivo=600_000)
        d.update(kw)
        return MaintenanceTask(**d)

    req = PlanRequest(
        t0=date(2026, 6, 1), horizonte_dias=150,
        tasks=[task("C-11", asset_id=13, riesgo_fallo_base=0.20),
               task("C-15", asset_id=7, duracion_dias=3),
               task("C-34", asset_id=19, requiere_jackup=True, coste_intervencion=250_000)],
        periodos_ambientales=periodos, n_escenarios_meteo=30, weather_provider="synthetic",
        seed=42, vetos_duros=[VetoDuro(alcance="jackup", mes_inicio=6, mes_fin=8,
                                       motivo="reproduccion marsopa jun-ago (KB)")])

    print("resolviendo (NSGA-II 3 semillas, greedy y exact_check), tarda unos segundos")
    front = solve_plan(req)

    # [1] frente y knee
    n = len(front.soluciones)
    check("front.no_vacio", n >= 1, f"{n} soluciones no dominadas | {front.request_resumen}")
    check("front.knee", front.knee_idx is not None and 0 <= front.knee_idx < n,
          f"knee_idx={front.knee_idx}")

    F = np.array([_obj(s) for s in front.soluciones])
    g = _obj(front.baseline_greedy)

    # [2] alguna solucion del frente domina (debilmente) al greedy
    tol = 1e-6
    domina = np.any(np.all(F <= g + tol, axis=1) & np.any(F < g - tol, axis=1)) or \
        np.any(np.all(np.abs(F - g) <= tol, axis=1))
    check("front.domina_greedy", bool(domina),
          f"greedy coste={g[0]:,.0f} riesgo={g[1]:.3f} amb={g[2]:.1f}; "
          f"mejor coste del frente={F[:,0].min():,.0f}")

    # [3] trade-off real: la solucion de min-coste difiere de la de min-impacto ambiental
    i_coste, i_amb = int(F[:, 0].argmin()), int(F[:, 2].argmin())
    trade = i_coste != i_amb and (F[i_coste, 2] > F[i_amb, 2] + tol)
    check("front.trade_off", bool(trade),
          f"min-coste amb={F[i_coste,2]:.1f} vs min-amb amb={F[i_amb,2]:.1f} "
          f"(coste {F[i_coste,0]:,.0f} frente a {F[i_amb,0]:,.0f})")

    # [4] validacion exacta
    ve = front.validacion_exacta
    check("front.exact_check", ve is not None and ve.optimo_alcanzado,
          f"{ve.detalle}" if ve else "sin validacion")

    # informe del knee
    knee = front.soluciones[front.knee_idx]
    dias = ", ".join(f"{s.task_id}@d{s.dia_inicio}" for s in knee.schedule)
    ko = knee.objetivos
    print(f"\nknee: [{dias}] coste={ko.coste_esperado:,.0f} riesgo={ko.riesgo_esperado:.3f} "
          f"impacto={ko.impacto_ambiental:.1f}")

    if FAILS:
        print(f"SMOKE F3: FALLOS {FAILS}")
        sys.exit(1)
    print("SMOKE F3: TODO PASS")


if __name__ == "__main__":
    main()
