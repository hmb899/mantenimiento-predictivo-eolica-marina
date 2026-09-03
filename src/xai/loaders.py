"""
cargadores verificables para los notebooks XAI

todo lo que se carga aqui tiene que reproducir exactamente los numeros committeados. la
verificacion vive en scripts/verify_xai_loaders.py y es el gate de la fase, asi que no hay que
fiarse de una figura cuyo cargador no haya pasado. farm es siempre "B" o "C".

pesos y artefactos versionados en git, como excepcion al .gitignore
    AE-C deployed (farm_c_best.pt)
    AE-B honesto s42 (farm_b_ident_s42_best.pt), el deployed original se perdio
    CRGN-C top100 s42
    TrajDev C/B s42 (state_dicts, z_stats, scaler y ch_names)
    scaler y pca de las dos granjas, regenerados con verificacion delta 0.0

load_scores
    los scores por ventana de cualquier detector salen del registro DETECTORS de fuse_multi,
    fuente unica de verdad de rutas y claves npz
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
# los notebooks importan esto sin instalar el paquete, de ahi el sys.path a mano
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

M = ROOT / "results" / "models"
P = ROOT / "data" / "processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# meta y scores
def load_meta(farm: str) -> dict[int, dict]:
    """meta canonica de eventos de la granja, labels y timestamps invariantes a la semilla."""

    fl = farm.lower()
    raw = json.load(open(M / f"ae_scores_farm_{fl}_meta.json", encoding="utf-8"))
    out = {}
    # las claves llegan como cadena del json y aqui pasan a int, que es como las usa el resto
    for eid, m in raw.items():
        out[int(eid)] = {"event_label": m["event_label"], "asset_id": int(m["asset_id"]),
                         "labels": np.asarray(m["labels"], np.int8),
                         "window_timestamps": pd.to_datetime(m["window_timestamps"])}
    return out


def load_scores(farm: str, detector: str, seed: int = 42) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """scores por ventana ya committeados de un detector, train y por evento, via el registro
    de fuse_multi. devuelve (train, {eid: scores})."""

    import fuse_multi  # lazy, arrastra run_model y torch

    fl = farm.lower()
    meta = load_meta(farm)
    # el orden de eids es el mismo en la peticion y en la respuesta, y de ahi sale el zip
    eids = list(meta.keys())
    train, evs = fuse_multi.load_detector(fl, detector, seed, [str(e) for e in eids])
    return train, {e: s for e, s in zip(eids, evs)}


# TrajDev
def load_trajdev(farm: str, seed: int = 42) -> dict:
    """ensemble TrajDev (s42 es el de campana, verificado exacto), stats leakage-free e info de
    la representacion. seed != 42 carga un bundle re-entrenado con el mismo comando que
    save42 y otra semilla, para el multiseed del triaje; solo C tiene bundles multi-seed y B
    sigue en s42."""

    from src.trajdev.model import TrajForecaster

    ckpt = M / (f"trajdev3_model_farm_c_s{seed}.pt" if farm == "C" else f"trajdevb_model_farm_b_s{seed}.pt")
    d = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    a = d["arch"]
    models = []
    # el bundle guarda un state_dict por miembro, la arquitectura es comun a todos
    for sd in d["state_dicts"]:
        m = TrajForecaster(a["k"], a["context"], a["horizon"], hidden=a["hidden"], layers=a["layers"])
        m.load_state_dict(sd)
        m.to(DEVICE).eval()
        models.append(m)
    ch_names = list(d["ch_names"]) if d.get("ch_names") else None
    masks = {}
    if ch_names:
        # los canales de valores llevan prefijo val__, y solo se guarda la mascara no vacia
        arrs = {"val": np.array([c.startswith("val__") for c in ch_names]),
                "all": np.ones(len(ch_names), dtype=bool)}
        masks = {k: v for k, v in arrs.items() if v.any()}
    return {"farm": farm, "models": models, "arch": a, "z_stats": d["z_stats"],
            "scaler": d.get("scaler"), "sub_h": d["sub_h"], "ch_names": ch_names,
            "masks": masks, "ckpt": str(ckpt)}


def load_trajdev_events(farm: str) -> list[dict]:
    """ventanas de eventos en el espacio de TrajDev, las mismas que la campana. B son las
    curated de 30 feats normalizadas con su scaler guardado, C el cache atlas3 tal cual."""

    if farm == "C":
        from signal_atlas import load_cache

        _tw, _ch, _pairs, seqs = load_cache("3")
        return [{"event_id": int(s["event_id"]), "event_label": s["event_label"],
                 "asset_id": int(s["asset_id"]), "labels": np.asarray(s["labels"], np.int8),
                 "window_timestamps": pd.to_datetime(np.asarray(s["window_timestamps"]).ravel()),
                 "data": np.asarray(s["windows"], np.float32)} for s in seqs]
    bundle = load_trajdev("B")
    mu, sd = bundle["scaler"]["mu"], bundle["scaler"]["sd"]
    meta = json.load(open(P / "farm_b_test_metadata.json", encoding="utf-8"))
    zt = np.load(P / "farm_b_test_sequences.npz", allow_pickle=True)
    out = []
    for eid, m in meta.items():
        # normaliza in place con el scaler del propio bundle, para no duplicar el array
        x = zt[eid].astype(np.float32)
        x -= mu
        x /= sd
        out.append({"event_id": int(eid), "event_label": m["event_label"],
                    "asset_id": int(m["asset_id"]), "labels": np.asarray(m["labels"], np.int8),
                    "window_timestamps": pd.to_datetime(m["window_timestamps"]), "data": x})
    return out


def trajdev_forecast_errors(bundle: dict, data: np.ndarray, batch: int = 4096):
    """(err, dis) del ensemble sobre ventanas (N, W, K), err (N, H, K) con signo y dis
    desacuerdo."""

    from src.trajdev.model import predict_windows

    a = bundle["arch"]
    return predict_windows(bundle["models"], data, a["context"], a["horizon"], DEVICE)


def trajdev_channel_z(bundle: dict, err: np.ndarray) -> dict[str, np.ndarray]:
    """familia z por canal (N, K) con las stats de train (leakage-free), la atribucion nativa
    de TrajDev. devuelve z (nivel), z_slope (tasa) y z_pers (persistencia direccional)."""

    zs = bundle["z_stats"]
    a_mean = np.abs(err).mean(axis=1)  # nivel, error medio sobre el horizonte
    a_slope = np.abs(err)[:, -1, :] - np.abs(err)[:, 0, :]  # tasa, ultimo paso menos primero
    # el valor absoluto va fuera de la media, asi un error que siempre apunta al mismo lado no
    # se cancela y el que oscila si, que es lo que mide la persistencia
    a_pers = np.abs(err.mean(axis=1))
    return {"z": (a_mean - zs["mu_c"]) / zs["sd_c"],
            "z_slope": (a_slope - zs["mu_s"]) / zs["sd_s"],
            "z_pers": (a_pers - zs["mu_p"]) / zs["sd_p"]}


def trajdev_score(bundle: dict, z: np.ndarray, variant: str = "dep_zmax_val") -> np.ndarray:
    """agrega el z por canal al score escalar de la variante, las de los gates committeados."""

    farm = bundle["farm"]
    zv = z[:, bundle["masks"]["val"]] if (farm == "C" and "val" in bundle["masks"]) else z
    if variant == "dep_zmax_val":
        return zv.max(axis=1)
    if variant == "dep_z_top3":
        # top-k en B se define sobre todos los canales (30 = val); en C el gate usa z global
        base = z
        return np.sort(base, axis=1)[:, -3:].mean(axis=1)
    if variant == "dep_z":
        return z.mean(axis=1)
    raise ValueError(f"variante no soportada: {variant}")


# AE
def load_ae(farm: str) -> dict:
    """AE para XAI, con el mapa afin sensor a K congelado y los nombres de sensores.

    en las dos granjas el AE explicado a nivel de pesos es la cadena reproducible, porque los
    artefactos del deployed original se perdieron. B es farm_b_ident_s42_best.pt (0.7213) con
    WindowScaler y WindowPCA regenerados (delta 0.0); C es farm_c_ident_s42_best.pt (0.7432,
    nivel deployed) con el mapa afin de la linea-cache recuperado por regresion
    (farm_c_sensor2k.npz, residuo 1.5e-4), porque alli el deploy y el cache son dos fits
    distintos y solo el segundo se reproduce. los numeros del campeon y de las fusiones siguen
    saliendo de los dumps.

    el mapa se expone siempre como afin generico, k = raw @ W + b con W de F x K, e inverse()
    usa la pseudo-inversa, la preimagen de minimos cuadrados, que coincide con la inversa de
    PCA cuando el mapa es scaler+PCA.
    """

    from src.models.bilstm_autoencoder import BiLSTMAutoencoder

    # k es el numero de componentes con el que se entreno el AE de cada granja
    if farm == "C":
        # en C el mapa sensor a componente viene ya ajustado en un npz
        ckpt, k = M / "farm_c_ident_s42_best.pt", 55
        z = np.load(M / "farm_c_sensor2k.npz", allow_pickle=True)
        W, b = np.asarray(z["W"], np.float64), np.asarray(z["b"], np.float64)
        feats = [str(c) for c in z["feats"]]
    else:
        # en B se compone a mano a partir de los dos objetos ajustados
        ckpt, k = M / "farm_b_ident_s42_best.pt", 18
        # WindowScaler (global standard)
        scaler = pickle.load(open(M / "farm_b_scaler.pkl", "rb"))
        pca = pickle.load(open(M / "farm_b_pca.pkl", "rb"))  # WindowPCA (K=18)
        zf = np.load(P / "farm_b_train_windows.npz", allow_pickle=True)
        feats = [str(c) for c in zf["feature_cols"]]
        comp, pmean = pca.pca.components_, pca.pca.mean_
        g = scaler._global_scaler
        # composicion afin exacta de standard-scaler y PCA, k = ((x - m)/s - pm) @ C^T
        W = (comp / np.asarray(g.scale_)[None, :]).T  # (F, K)
        b = ((-np.asarray(g.mean_) / np.asarray(g.scale_)) - pmean) @ comp.T

    Wpinv = np.linalg.pinv(W)  # (K, F)

    model = BiLSTMAutoencoder(input_size=k, encoder_units=[64, 16], decoder_units=[16, 64],
                              dropout=0.1, out_activation="linear")
    # strict=False tolera que el checkpoint traiga claves que esta arquitectura no declara
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    model.to(DEVICE).eval()

    def transform(raw_windows: np.ndarray) -> np.ndarray:
        """de (N, W, F) en espacio sensor a (N, W, K) en espacio del AE, afin por filas."""

        # se aplana a filas porque el mapa es por timestep, no por ventana
        n, w, f = raw_windows.shape
        flat = raw_windows.reshape(-1, f).astype(np.float64)
        return (flat @ W + b).reshape(n, w, -1).astype(np.float32)

    def inverse(k_windows: np.ndarray) -> np.ndarray:
        """de (N, W, K) a (N, W, F), la preimagen de minimos cuadrados, la inversa de
        PCA+scaler."""

        n, w, kk = k_windows.shape
        flat = k_windows.reshape(-1, kk).astype(np.float64)
        return ((flat - b) @ Wpinv).reshape(n, w, -1).astype(np.float32)

    return {"farm": farm, "model": model, "k": k, "feats": feats, "transform": transform,
            "inverse": inverse, "ckpt": str(ckpt),
            "affine": {"W": W, "b": b, "Wpinv": Wpinv}}


def load_ae_events(farm: str) -> list[dict]:
    """ventanas de eventos en espacio sensor, para atribucion, con las longitudes de la
    meta."""

    fl = farm.lower()
    if farm == "C":
        npz, meta_p = P / "farm_c_xai_test_sequences.npz", P / "farm_c_xai_test_metadata.json"
    else:
        npz, meta_p = P / "farm_b_test_sequences.npz", P / "farm_b_test_metadata.json"
    meta = json.load(open(meta_p, encoding="utf-8"))
    z = np.load(npz, allow_pickle=True)
    out = []
    for eid, m in meta.items():
        out.append({"event_id": int(eid), "event_label": m["event_label"],
                    "asset_id": int(m["asset_id"]), "labels": np.asarray(m["labels"], np.int8),
                    "window_timestamps": pd.to_datetime(m["window_timestamps"]),
                    "raw": z[eid].astype(np.float32)})
    return out


@torch.no_grad()
def ae_scores(bundle: dict, k_windows: np.ndarray, batch: int = 1024) -> np.ndarray:
    """MAE por ventana del AE cargado, el score oficial mae_window."""

    out = []
    for s in range(0, len(k_windows), batch):
        x = torch.tensor(k_windows[s:s + batch], dtype=torch.float32, device=DEVICE)
        # media sobre pasos y componentes a la vez, una cifra por ventana
        out.append((x - bundle["model"](x)).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)
