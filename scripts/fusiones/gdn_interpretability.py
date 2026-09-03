"""interpretabilidad de GDN, el grafo de sensores y los embeddings aprendidos.

es lo unico que GDN tiene y los demas no. el AE se explica con SHAP post-hoc y CRGN lleva un
grafo Spearman fijo; solo GDN aprende la estructura sensor-sensor, asi que aqui se abre.

carga un checkpoint ya entrenado, gdn_farm_{f}_{features}_s42.pt, reajusta la normalizacion
del scorer sobre la cola normal de validacion, con el mismo split por seed que run_full, y
produce por granja el set core mas el extended.

    core
        1 grafo aprendido      model.adjacency(), red, heatmap y centralidad con hubs
        2 embeddings de sensor model.node_embeddings(), t-SNE y heatmap de similitud coseno
        3 localizacion evento  (err - mediana) / IQR por sensor, heatmap sensor x tiempo
                               y root-cause
        4 importancia global   desviacion en anomalia frente a normal, que sensores guian
                               la deteccion
    extended
        5 comunidades del grafo por greedy modularity, coloreando la red y el t-SNE, que
                               enfrenta la estructura con el embedding
        6 ablacion del grafo   aprendido frente a Spearman fijo y frente a sin grafo, que la
                               adyacencia es inyectable

hooks verificados en src/gdn. model.node_embeddings() da (K,d); model.adjacency() da (K,K),
donde A[i,j] es la arista de j a i; GDNScorer._sensor_errors(w) con .median_ y .iqr_ da la
desviacion normalizada (N,K); y feats son los sensores.

    python scripts/fusiones/gdn_interpretability.py --farm B
    python scripts/fusiones/gdn_interpretability.py --farm C
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, donde vive run_gdn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raiz del repo (from src.X)

import run_gdn
from src.gdn.graph import cosine_similarity_matrix
from src.gdn.scorer import GDNScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FIG_DIR = Path("results/figures/gdn")
TAB_DIR = Path("results/tables")


def short(name: str) -> str:
    """etiqueta corta y unica de sensor para las figuras, quita el _avg por defecto, abrevia
    los prefijos y conserva _min, _max y _std para no provocar colisiones."""

    # sensor_53_avg queda en s53, y wind_speed_59_min en ws59_min
    s = name.replace("_avg", "")
    return (s.replace("reactive_power_", "rp").replace("wind_speed_", "ws")
            .replace("power_", "pw").replace("sensor_", "s"))


# carga
def load_trained(farm: str, seed: int = 42):
    """reproduce el estado de run_full sin reentrenar y devuelve cfg, model, scorer, feats,
    test_seqs y val_norm, con el split de val por seed y la normalizacion ya ajustada."""

    cfg = run_gdn.load_farm_config(farm)
    features = cfg["features"]
    scaled, test_seqs, feats = run_gdn.load_representation(farm, cfg, features, seed)
    w = scaled.shape[1]
    horizon = cfg["windowing"].get("horizon", 1)
    # el split de val se rehace con la misma semilla que run_full, para caer en las mismas
    # ventanas normales sobre las que se calibro el scorer
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(scaled))
    n_val = int(round(len(scaled) * cfg["training"]["val_ratio"]))
    val_norm = scaled[perm[:n_val]]
    tag = f"gdn_farm_{farm.lower()}_{features}_s{seed}"
    ck = torch.load(Path(cfg["output"]["models_dir"]) / f"{tag}.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = run_gdn.build_model(cfg, n_nodes=ck["n_nodes"], context=ck["context"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sc = cfg["scoring"]
    scorer = GDNScorer(model, context=ck["context"], device=device, aggregation=sc["aggregation"],
                       topk_q=sc["topk_q"], exo_idx=None, batch_size=sc["batch_size"],
                       percentile=cfg["threshold"]["percentile"], horizon=horizon)
    scorer.fit_normalization(val_norm)
    feats = [str(f) for f in ck["feats"]]
    logger.info(f"cargado {tag}: {len(feats)} sensores, {len(test_seqs)} eventos, val {val_norm.shape}, {device}")
    return cfg, model, scorer, feats, test_seqs, val_norm


def norm_dev(scorer: GDNScorer, windows: np.ndarray) -> np.ndarray:
    """desviacion normalizada por sensor (N, K) = (error_forecast - mediana_val) / IQR_val."""
    return (scorer._sensor_errors(windows) - scorer.median_) / scorer.iqr_


def learned_digraph(adj: np.ndarray) -> nx.DiGraph:
    """DiGraph desde la adyacencia, donde A[i,j]=1 es la arista de j a i, o sea que i recibe
    de j. sin self-loops."""
    g = nx.DiGraph()
    g.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if i != j and adj[i, j] > 0:
                g.add_edge(j, i)
    return g


def communities(g: nx.DiGraph) -> dict[int, int]:
    """comunidades por modularidad greedy sobre la version no dirigida, de nodo a comunidad."""
    try:
        comms = nx.community.greedy_modularity_communities(g.to_undirected())
        return {n: c for c, nodes in enumerate(comms) for n in nodes}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"comunidades fallback (1 sola): {e}")
        return dict.fromkeys(g.nodes(), 0)


# core 1, grafo
def fig_graph(adj: np.ndarray, feats: list[str], comm: dict[int, int], farm: str) -> dict:
    g = learned_digraph(adj)
    K = adj.shape[0]
    out_deg = dict(g.out_degree())  # a cuantos sensores predice j, o sea su influencia
    in_deg = dict(g.in_degree())  # de cuantos depende i
    pr = nx.pagerank(g) if g.number_of_edges() else dict.fromkeys(range(K), 1 / K)
    # la betweenness es cuadratica en el numero de nodos, asi que por encima de 120 se anula
    btw = nx.betweenness_centrality(g) if K <= 120 else dict.fromkeys(range(K), 0.0)

    # tabla de centralidad
    rows = [{"sensor": feats[i], "community": comm.get(i, 0), "out_degree": out_deg.get(i, 0),
             "in_degree": in_deg.get(i, 0), "pagerank": round(pr.get(i, 0.0), 5),
             "betweenness": round(btw.get(i, 0.0), 5)} for i in range(K)]
    rows.sort(key=lambda r: (r["out_degree"], r["pagerank"]), reverse=True)
    _write_csv(TAB_DIR / f"gdn_graph_centrality_farm_{farm.lower()}.csv", rows)

    # red dibujada (color=comunidad, tamano~out_degree); etiquetas solo a hubs en C
    pos = nx.spring_layout(g, seed=42, k=1.5 / np.sqrt(max(K, 1)))
    ncolor = [comm.get(i, 0) for i in range(K)]
    nsize = [80 + 60 * out_deg.get(i, 0) for i in range(K)]
    plt.figure(figsize=(13, 11))
    nx.draw_networkx_edges(g, pos, alpha=0.18, width=0.5, arrows=False)
    nx.draw_networkx_nodes(g, pos, node_color=ncolor, node_size=nsize, cmap="tab20", alpha=0.9)
    # en B se etiquetan los 30 sensores; en C solo los 18 primeros, o la red no se lee
    hubs = [r["sensor"] for r in rows[:(K if farm == "B" else 18)]]
    labels = {i: short(feats[i]) for i in range(K) if feats[i] in hubs}
    nx.draw_networkx_labels(g, pos, labels, font_size=8)
    plt.title(f"GDN grafo aprendido de sensores - Granja {farm} "
              f"({K} nodos, top-k coseno; color=comunidad, tamano~out-degree)")
    plt.axis("off")
    _save(FIG_DIR / f"gdn_graph_farm_{farm.lower()}.png")

    # heatmap de adyacencia ordenado por comunidad
    order = sorted(range(K), key=lambda i: (comm.get(i, 0), -out_deg.get(i, 0)))
    plt.figure(figsize=(9, 8))
    plt.imshow(adj[np.ix_(order, order)], cmap="Greys", aspect="auto", interpolation="nearest")
    plt.title(f"GDN adyacencia aprendida (orden por comunidad) - Granja {farm}")
    plt.xlabel("sensor emisor j"); plt.ylabel("sensor receptor i")
    _save(FIG_DIR / f"gdn_adjacency_heatmap_farm_{farm.lower()}.png")

    # barras de hubs (out-degree)
    top = rows[:min(20, K)]
    plt.figure(figsize=(10, 5))
    plt.bar([short(r["sensor"]) for r in top], [r["out_degree"] for r in top], color="steelblue")
    plt.title(f"GDN sensores hub (out-degree = a cuantos predice) - Granja {farm}")
    plt.ylabel("out-degree"); plt.xticks(rotation=60, ha="right", fontsize=8)
    _save(FIG_DIR / f"gdn_hubs_farm_{farm.lower()}.png")

    n_comm = len(set(comm.values()))
    logger.info(f"[{farm}] grafo: {g.number_of_edges()} aristas, {n_comm} comunidades, "
                f"hub top = {top[0]['sensor']} (out-degree {top[0]['out_degree']})")
    return {"n_edges": g.number_of_edges(), "n_communities": n_comm,
            "top_hubs": [r["sensor"] for r in rows[:5]], "centrality_rows": rows}


# core 2, embeddings
def fig_embeddings(emb: np.ndarray, feats: list[str], comm: dict[int, int], farm: str) -> None:
    K = emb.shape[0]
    # la perplejidad de t-SNE escala con K pero se acota a [5, 30], que B trae 30 sensores
    perp = float(min(30, max(5, (K - 1) // 3)))
    xy = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42,
              learning_rate="auto").fit_transform(emb)
    plt.figure(figsize=(11, 9))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=[comm.get(i, 0) for i in range(K)], cmap="tab20", s=70, alpha=0.9)
    for i in range(K):
        # en B caben todas las etiquetas; en C se anota una de cada K/40
        if farm == "B" or i % max(1, K // 40) == 0:
            plt.annotate(short(feats[i]), (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.8)
    plt.title(f"GDN embeddings de sensor (t-SNE, perp={perp:.0f}; color=comunidad del grafo) - Granja {farm}")
    plt.colorbar(sc, label="comunidad")
    _save(FIG_DIR / f"gdn_embeddings_tsne_farm_{farm.lower()}.png")

    sim = cosine_similarity_matrix(torch.tensor(emb)).numpy()
    out_order = sorted(range(K), key=lambda i: comm.get(i, 0))
    plt.figure(figsize=(9, 8))
    plt.imshow(sim[np.ix_(out_order, out_order)], cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.title(f"GDN similitud coseno de embeddings (orden por comunidad) - Granja {farm}")
    plt.colorbar(label="cos sim"); plt.xlabel("sensor"); plt.ylabel("sensor")
    _save(FIG_DIR / f"gdn_embedding_similarity_farm_{farm.lower()}.png")
    logger.info(f"[{farm}] embeddings t-SNE + similitud guardados")


# core 3 y 4, localizacion e importancia
def event_localization(scorer: GDNScorer, feats: list[str], test_seqs: list[dict], farm: str) -> dict:
    K = len(feats)
    anomalies, all_dev_an, all_dev_nm, summary = [], [], [], []
    for s in test_seqs:
        dev = norm_dev(scorer, s["windows"])  # (n_win, K)
        labels = np.asarray(s["labels"], int)
        is_an = s["event_label"] == "anomaly"
        if labels.any():
            all_dev_an.append(dev[labels == 1])
        if (labels == 0).any():
            all_dev_nm.append(dev[labels == 0])
        region = dev[labels == 1] if labels.any() else dev  # ventanas de fallo, o todas
        peak_per_sensor = region.max(axis=0)  # (K,), pico por sensor dentro del fallo
        top = np.argsort(peak_per_sensor)[::-1][:3]
        summary.append({"event_id": s["event_id"], "label": s["event_label"], "asset_id": int(s["asset_id"]),
                        "n_windows": int(len(dev)), "n_anomaly_windows": int(labels.sum()),
                        "top1_sensor": feats[top[0]], "top1_dev": round(float(peak_per_sensor[top[0]]), 3),
                        "top2_sensor": feats[top[1]], "top2_dev": round(float(peak_per_sensor[top[1]]), 3),
                        "top3_sensor": feats[top[2]], "top3_dev": round(float(peak_per_sensor[top[2]]), 3)})
        if is_an:
            anomalies.append((s, dev, labels, float(peak_per_sensor.max())))
    _write_csv(TAB_DIR / f"gdn_rootcause_summary_farm_{farm.lower()}.csv", summary)

    # figura multipanel con los hasta 6 eventos de anomalia de mayor pico de desviacion
    anomalies.sort(key=lambda t: t[3], reverse=True)
    sel = anomalies[:6]
    if sel:
        ncol = min(3, len(sel)); nrow = int(np.ceil(len(sel) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
        for ax, (s, dev, labels, _) in zip(axes.ravel(), sel):
            top_s = np.argsort(dev.max(axis=0))[::-1][:12]  # 12 sensores mas desviados
            # el vmax al p99 evita que un pico aislado aplaste la escala de color
            im = ax.imshow(dev[:, top_s].T, aspect="auto", cmap="magma", interpolation="nearest",
                           vmin=0, vmax=np.percentile(dev, 99))
            ax.set_yticks(range(len(top_s))); ax.set_yticklabels([short(feats[i]) for i in top_s], fontsize=7)
            if labels.any():
                on = int(np.argmax(labels))  # inicio de la zona etiquetada
                ax.axvline(on, color="cyan", ls="--", lw=1.2)
            ax.set_title(f"ev{s['event_id']} (turbina {s['asset_id']})", fontsize=9)
            ax.set_xlabel("ventana (tiempo)")
        for ax in axes.ravel()[len(sel):]:
            ax.axis("off")
        fig.suptitle(f"GDN localizacion por evento: desviacion normalizada sensor x tiempo "
                     f"(linea = inicio de anomalia) - Granja {farm}", fontsize=11)
        fig.colorbar(im, ax=axes.ravel().tolist(), label="desviacion (sigmas-IQR)", shrink=0.6)
        _save(FIG_DIR / f"gdn_event_localization_farm_{farm.lower()}.png", tight=False)

    # importancia global, media de desviacion en ventanas de anomalia frente a normales
    an = np.concatenate(all_dev_an) if all_dev_an else np.zeros((1, K))
    nm = np.concatenate(all_dev_nm) if all_dev_nm else np.zeros((1, K))
    imp = [{"sensor": feats[i], "mean_dev_anomaly": round(float(an[:, i].mean()), 4),
            "mean_dev_normal": round(float(nm[:, i].mean()), 4),
            "gap": round(float(an[:, i].mean() - nm[:, i].mean()), 4)} for i in range(K)]
    imp.sort(key=lambda r: r["gap"], reverse=True)
    _write_csv(TAB_DIR / f"gdn_sensor_importance_farm_{farm.lower()}.csv", imp)
    top = imp[:min(20, K)]
    x = np.arange(len(top))
    plt.figure(figsize=(11, 5))
    plt.bar(x - 0.2, [r["mean_dev_anomaly"] for r in top], 0.4, label="anomalia", color="crimson")
    plt.bar(x + 0.2, [r["mean_dev_normal"] for r in top], 0.4, label="normal", color="grey")
    plt.xticks(x, [short(r["sensor"]) for r in top], rotation=60, ha="right", fontsize=8)
    plt.title(f"GDN importancia de sensor (desviacion media en anomalia vs normal) - Granja {farm}")
    plt.ylabel("desviacion normalizada media"); plt.legend()
    _save(FIG_DIR / f"gdn_sensor_importance_farm_{farm.lower()}.png")

    n_an = sum(1 for s in test_seqs if s["event_label"] == "anomaly")
    logger.info(f"[{farm}] localizacion: {n_an} eventos de anomalia, top sensor global = {imp[0]['sensor']} "
                f"(gap {imp[0]['gap']})")
    return {"n_anomaly_events": n_an, "top_rootcause_sensors": [r["sensor"] for r in imp[:5]],
            "rootcause_summary": summary, "importance_rows": imp}


# extended, ablacion de grafo
def graph_ablation(cfg: dict, model, scorer: GDNScorer, feats: list[str], test_seqs: list[dict],
                   val_norm: np.ndarray, farm: str) -> dict:
    """ablacion del grafo, el aprendido frente al Spearman fijo y frente a la matriz llena,
    que es lo que dice si la estructura aprendida aporta."""

    import run_model
    from src.care_official.adapter import OfficialCAREAdapter
    from src.gdn.graph import spearman_similarity, topk_adjacency
    K, device = len(feats), scorer.device
    learned = model.adjacency().detach()
    spear = topk_adjacency(torch.tensor(spearman_similarity(val_norm[:, -1, :]), dtype=torch.float32),
                           model.topk, include_self=True).to(device)
    full = torch.ones(K, K, device=device)
    # el window_size del evaluador oficial sale de la propia representacion, no de la config
    w = test_seqs[0]["windows"].shape[1]
    adapter = OfficialCAREAdapter(raw_path=cfg["data"]["raw_path"], farm=farm, window_size=w)
    res = {}
    # la adyacencia entra por scorer._adj_override y acaba en model.forward(x, adjacency=...)
    # 'aprendido' tiene que reproducir el baseline de GDN, y ese es el gate de la ablacion
    for name, adj in [("aprendido", learned), ("spearman_fijo", spear), ("sin_grafo_full", full)]:
        sc2 = GDNScorer(model, context=scorer.context, device=device, aggregation=scorer.aggregation,
                        topk_q=scorer.topk_q, exo_idx=None, batch_size=scorer.batch_size,
                        percentile=scorer.percentile, horizon=scorer.horizon)
        sc2._adj_override = adj
        # cada grafo tiene su escala de error, asi que la normalizacion se reajusta por grafo
        sc2.fit_normalization(val_norm)
        # la evaluacion pasa por el harness oficial, el mismo del resto del proyecto
        ev = run_model.evaluate(sc2, val_norm, test_seqs, adapter, 42)
        res[name] = {"roc_ds": round(float(ev["best"]["roc_ds"]), 4), "care_cv": round(float(ev["cv"]["care"]), 4)}
        logger.info(f"[{farm}] ablacion grafo {name}: ROC-ds {res[name]['roc_ds']} / CARE {res[name]['care_cv']}")
    names = list(res)
    x = np.arange(len(names))
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - 0.2, [res[n]["roc_ds"] for n in names], 0.4, label="ROC-ds", color="teal")
    plt.bar(x + 0.2, [res[n]["care_cv"] for n in names], 0.4, label="CARE-CV", color="orange")
    plt.xticks(x, names); plt.ylabel("metrica"); plt.legend()
    plt.title(f"GDN ablacion del grafo (aprendido vs fijo vs sin) - Granja {farm}")
    _save(FIG_DIR / f"gdn_graph_ablation_farm_{farm.lower()}.png")
    return res


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def _save(path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  fig -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GDN interpretabilidad (Hito 6)")
    ap.add_argument("--farm", choices=["B", "C"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg, model, scorer, feats, test_seqs, val_norm = load_trained(args.farm, args.seed)
    adj = model.adjacency().detach().cpu().numpy()
    emb = model.node_embeddings().detach().cpu().numpy()
    comm = communities(learned_digraph(adj))

    g_info = fig_graph(adj, feats, comm, args.farm)
    fig_embeddings(emb, feats, comm, args.farm)
    ev_info = event_localization(scorer, feats, test_seqs, args.farm)
    # la ablacion es extended y no debe tumbar el core, de ahi el except amplio
    try:
        ablation = graph_ablation(cfg, model, scorer, feats, test_seqs, val_norm, args.farm)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{args.farm}] ablacion de grafo fallo (extended): {e}")
        ablation = {}

    summary = {"farm": args.farm, "seed": args.seed, "n_sensors": len(feats),
               "graph": {k: g_info[k] for k in ("n_edges", "n_communities", "top_hubs")},
               "events": {k: ev_info[k] for k in ("n_anomaly_events", "top_rootcause_sensors")},
               "graph_ablation": ablation}
    out = TAB_DIR / f"gdn_interpretability_farm_{args.farm.lower()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"[{args.farm}] resumen -> {out}")
    logger.info(f"=== GDN interpretabilidad Granja {args.farm} LISTA: "
                f"{g_info['n_communities']} comunidades, hubs {g_info['top_hubs'][:3]}, "
                f"root-cause {ev_info['top_rootcause_sensors'][:3]} ===")


if __name__ == "__main__":
    main()
