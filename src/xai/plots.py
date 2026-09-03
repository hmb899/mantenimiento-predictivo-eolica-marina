"""
figuras estandar de los notebooks XAI

siguen las convenciones de los papers SCADA/PdM, con estilo sobrio, sin
emojis, en castellano y con paleta colorblind-safe.

plot_event_timeline
    score y CUSUM con el tramo del evento sombreado
plot_channel_heatmap
    heatmap canal por tiempo del z por canal
plot_sensor_ranking
    ranking de sensores en barras horizontales
plot_member_stack
    descomposicion por miembro de la fusion
plot_coverage_matrix
    matriz de cobertura detector por evento
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# el orden fija el color de cada serie, y el 0 es el color de linea por defecto
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


# se llama una vez al principio del cuaderno, cambia los rcParams del proceso
def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def _shade_anomaly(ax, ts: pd.DatetimeIndex, labels: np.ndarray) -> None:
    """sombrea los tramos etiquetados como anomalia, la ventana CARE del evento. el naranja no
    es preferencia de estilo, es el color con el que se renderizaron las 13 figuras sombreadas
    embebidas en los cuadernos 01-04, asi que cambiarlo obliga a re-ejecutar los cuatro
    cuadernos y a comprobar que las figuras salen iguales."""

    lab = np.asarray(labels).astype(bool)
    if not lab.any():
        return
    # los ceros de guarda a los lados hacen que diff marque todo arranque y todo final de
    # tramo, y salen en pares, el par abre y el impar cierra
    edges = np.flatnonzero(np.diff(np.concatenate(([0], lab.astype(int), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(ts[a], ts[min(b, len(ts) - 1)], color="#D55E00", alpha=0.15, lw=0)


def plot_event_timeline(ts: pd.DatetimeIndex, S: np.ndarray, h: float, labels: np.ndarray,
                        score: np.ndarray | None = None, alarm_idx: int | None = None,
                        title: str = "", ax=None):
    """figura universal NBM, CUSUM con umbral h, tramo anomalo sombreado y score crudo
    tenue."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.8))
    _shade_anomaly(ax, ts, labels)
    if score is not None:
        # eje derecho propio, el score crudo y el CUSUM no comparten escala
        ax2 = ax.twinx()
        ax2.plot(ts, score, color="#999999", lw=0.6, alpha=0.6, zorder=1)
        ax2.set_ylabel("score", color="#777777", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7, colors="#777777")
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
    # los zorder crecientes dejan el sombreado al fondo y la alarma encima de todo
    ax.plot(ts, S, color=PALETTE[0], lw=1.2, zorder=3, label="CUSUM")
    ax.axhline(h, color="#333333", ls="--", lw=0.9, zorder=2, label="umbral h")
    if alarm_idx is not None:
        ax.axvline(ts[alarm_idx], color="#D55E00", lw=1.4, zorder=4)
        ax.annotate("alarma", (ts[alarm_idx], ax.get_ylim()[1] * 0.92), color="#D55E00",
                    fontsize=8, ha="left", xytext=(4, 0), textcoords="offset points")
    ax.set_title(title, loc="left")
    ax.set_ylabel("CUSUM")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    return ax


def plot_channel_heatmap(ts: pd.DatetimeIndex, z: np.ndarray, ch_names: list[str], top: int = 20,
                         title: str = "", vmax: float | None = None, ax=None, order: str = "max"):
    """heatmap canal por tiempo del z por canal (N, K), el que y el cuando de la deteccion.
    selecciona los `top` canales por z max (order='max') o por z medio en el tramo anomalo."""

    zt = np.asarray(z)
    rank = zt.max(axis=0) if order == "max" else zt.mean(axis=0)
    idx = np.argsort(rank)[::-1][:top][::-1]  # peor arriba al dibujar invertido
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.28 * len(idx) + 1.2))
    # el percentil recorta la cola alta, si no un solo pico plancha toda la escala de color
    vmax = vmax or float(np.percentile(zt[:, idx], 99.5))
    # imshow no entiende de fechas, extent le da los limites del eje x en numero de dia
    im = ax.imshow(zt[:, idx].T, aspect="auto", cmap="inferno", vmin=0, vmax=vmax,
                   extent=[mdates.date2num(ts[0]), mdates.date2num(ts[-1]), -0.5, len(idx) - 0.5],
                   origin="lower", interpolation="nearest")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([ch_names[i] for i in idx], fontsize=7)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.set_title(title, loc="left")
    ax.grid(False)
    plt.colorbar(im, ax=ax, pad=0.01, label="z")
    return ax, idx


def plot_sensor_ranking(values: np.ndarray, names: list[str], top: int = 15, title: str = "",
                        xlabel: str = "contribucion (z medio en el tramo pre-fallo)", ax=None):
    """barras horizontales estilo contribution-plot/ARCANA."""

    # el segundo reverso deja el mayor arriba, porque barh dibuja de abajo a arriba
    idx = np.argsort(values)[::-1][:top][::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.3 * top + 0.8))
    ax.barh(range(len(idx)), values[idx], color=PALETTE[0])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return ax, idx


def plot_member_stack(ts: pd.DatetimeIndex, contribs: dict[str, np.ndarray], labels: np.ndarray,
                      title: str = "", ax=None):
    """descomposicion aditiva exacta de la fusion, area apilada w_i*z_i por miembro."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.8))
    _shade_anomaly(ax, ts, labels)
    names = list(contribs.keys())
    # el area apilada solo admite aportaciones no negativas, las negativas se recortan a cero
    vals = np.vstack([np.clip(contribs[n], 0, None) for n in names])
    ax.stackplot(ts, vals, labels=names, colors=PALETTE[:len(names)], alpha=0.85, lw=0)
    ax.legend(loc="upper left", fontsize=8, ncols=len(names))
    ax.set_title(title, loc="left")
    ax.set_ylabel("contribucion w·z (>=0)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    return ax


def plot_coverage_matrix(detected: pd.DataFrame, title: str = "", ax=None):
    """matriz detector por evento, la celda es el lead time en horas si detecta y NaN si no.
    detected es un DataFrame con index=detector, columns=event_id y values=lead_h."""

    if ax is None:
        _, ax = plt.subplots(figsize=(0.34 * detected.shape[1] + 2, 0.5 * detected.shape[0] + 1.2))
    data = detected.to_numpy(dtype=float)
    # enmascarar los NaN evita que el colormap los pinte, y debajo se marcan uno a uno
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(range(detected.shape[1]))
    ax.set_xticklabels(detected.columns, fontsize=7, rotation=90)
    ax.set_yticks(range(detected.shape[0]))
    ax.set_yticklabels(detected.index, fontsize=9)
    for (i, j), v in np.ndenumerate(data):
        if np.isnan(v):
            ax.text(j, i, "-", ha="center", va="center", color="#BBBBBB", fontsize=7)
    ax.set_title(title, loc="left")
    ax.grid(False)
    plt.colorbar(im, ax=ax, pad=0.01, label="lead time (h)")
    return ax
