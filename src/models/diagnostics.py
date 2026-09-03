"""
graficas de diagnostico para la evaluacion de la deteccion de anomalias, con un solo
punto de entrada publico

plot_score_vs_time
    error de reconstruccion frente al indice de ventana en una seleccion representativa
    de eventos, anomalias detectadas, anomalias no detectadas y eventos normales
    guarda un PNG bajo figures_dir/diagnostics/
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ventana de la media movil del suavizado visual, 18 muestras = 3h a 10 min
_SMOOTH_WINDOW = 18


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """aplica una media movil uniforme del tamano de ventana dado, devuelve una copia si
    la ventana es demasiado pequena."""

    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    # mode same conserva la longitud del array, a cambio los extremos quedan sesgados
    return np.convolve(arr, kernel, mode="same")


def _shade_gt_regions(ax, gt: np.ndarray, label: str = "GT anomalía") -> None:
    """sombrea sobre ax los tramos contiguos de gt==1."""

    in_region = False
    region_start = 0
    label_added = False
    # se recorre gt marcando donde arranca cada tramo y se pinta al volver a cero
    for i in range(len(gt)):
        if gt[i] == 1 and not in_region:
            region_start = i
            in_region = True
        elif gt[i] == 0 and in_region:
            # la etiqueta se pone solo en el primer tramo para no repetirla en la leyenda
            kw: dict = {"label": label} if not label_added else {}
            ax.axvspan(region_start, i, alpha=0.18, color="red", **kw)
            label_added = True
            in_region = False
    # un tramo que llega hasta el final del array se cierra aqui a mano
    if in_region:
        kw = {"label": label} if not label_added else {}
        ax.axvspan(region_start, len(gt), alpha=0.18, color="red", **kw)


def _select_events(test_sequences: list[dict], cusum_results: list[dict], max_events: int) -> list[tuple[dict, dict, str]]:
    """elige hasta max_events eventos representativos que cubran todos los resultados
    de deteccion."""

    pairs = list(zip(test_sequences, cusum_results))

    # cuatro grupos por resultado, anomalia con y sin alarma y normal con y sin alarma
    detected  = [(s, r) for s, r in pairs
                 if s["event_label"] == "anomaly" and r["alarm_raised"]]
    missed    = [(s, r) for s, r in pairs
                 if s["event_label"] == "anomaly" and not r["alarm_raised"]]
    normal_fp = [(s, r) for s, r in pairs
                 if s["event_label"] == "normal" and r["alarm_raised"]]
    normal_ok = [(s, r) for s, r in pairs
                 if s["event_label"] == "normal" and not r["alarm_raised"]]

    # los detectados se ordenan por posicion de alarma, primero la deteccion mas temprana
    # el 99999 manda al final a los que no traen indice de alarma
    detected.sort(key=lambda x: x[1]["alarm_window_idx"] if x[1]["alarm_window_idx"] is not None else 99999)

    # se rellena por prioridad, dos detecciones y luego un ejemplo de cada caso restante
    to_plot: list[tuple[dict, dict, str]] = []
    for s, r in detected[:2]:
        to_plot.append((s, r, "Anomalía detectada"))
    if len(to_plot) < max_events:
        for s, r in missed[:1]:
            to_plot.append((s, r, "Anomalía NO detectada"))
    if len(to_plot) < max_events:
        for s, r in normal_fp[:1]:
            to_plot.append((s, r, "Normal — falsa alarma"))
    if len(to_plot) < max_events:
        for s, r in normal_ok[:1]:
            to_plot.append((s, r, "Normal — sin alarma"))

    return to_plot[:max_events]


def plot_score_vs_time(farm: str, test_sequences: list[dict], cusum_results: list[dict], threshold: float,
                       figures_dir: str | Path, model_label: str = "AE", smooth_window: int = _SMOOTH_WINDOW,
                       max_events: int = 4) -> None:
    """dibuja el error de reconstruccion frente al indice de ventana en eventos
    representativos y lo guarda como PNG."""

    # matplotlib se importa dentro y con backend Agg, la figura se escribe sin display
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available skipping diagnostic plot")
        return

    to_plot = _select_events(test_sequences, cusum_results, max_events)
    if not to_plot:
        logger.warning("plot_score_vs_time: no events to plot skipping")
        return

    n = len(to_plot)
    # squeeze=False deja siempre un array 2D, por eso luego se indexa axes[:, 0]
    # el alto de la figura crece con el numero de eventos
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), squeeze=False)

    # un panel por evento, mismo umbral en todos
    for ax, (seq, res, kind) in zip(axes[:, 0], to_plot):
        errors = res["window_errors"]
        gt     = seq["labels"]
        x      = np.arange(len(errors))
        smooth = _moving_average(errors, smooth_window)

        # el score crudo va en tenue y el suavizado en trazo grueso
        ax.plot(x, errors, alpha=0.22, color="steelblue", linewidth=0.7)
        ax.plot(x, smooth, color="steelblue", linewidth=1.6,
                label=f"Score suavizado (MA-{smooth_window})")

        # umbral de deteccion
        ax.axhline(threshold, color="darkorange", linestyle="--",
                   linewidth=1.3, label=f"Umbral ({threshold:.4f})")

        # sombreado de la anomalia segun la verdad de referencia
        _shade_gt_regions(ax, gt)

        # marca de la alarma CUSUM
        if res["alarm_raised"] and res["alarm_window_idx"] is not None:
            ax.axvline(res["alarm_window_idx"], color="crimson",
                       linewidth=2.0, linestyle=":", label="Alarma CUSUM")

        ax.set_title(
            f"[{kind}]   Evento {seq['event_id']}",
            fontsize=10, fontweight="bold", loc="left",
        )
        ax.set_ylabel("Error MAE", fontsize=9)
        ax.set_xlabel("Ventana #", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.75)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Diagnóstico: Score vs Tiempo — Granja {farm}  [{model_label}]",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    # el nombre del PNG lleva granja y modelo, una ejecucion nueva pisa la anterior
    out_dir = Path(figures_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"farm_{farm.lower()}_{model_label.lower()}_score_vs_time.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"      Diagnostic plot saved to {out_path}")
