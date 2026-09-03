"""
two-phase training for TimeADDM

phase 1
    LSTMAE reconstruction on normal training windows
phase 2
    diffusion U-Net noise prediction on LSTMAE latents
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.trainer import EarlyStopping
from src.timeaddm.diffusion_utils import noise_estimation_loss

logger = logging.getLogger(__name__)


class EMA:
    """exponential moving average of the model weights, a standard anti-overfit trick
    for diffusion models."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # el backup guarda los pesos crudos mientras el modelo lleva los EMA
        self._backup: dict | None = None


    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)  # los buffers no float se copian sin promediar


    def apply_shadow(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)


    def restore(self, model: nn.Module) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None


def train_lstmae(model: nn.Module, train_windows: np.ndarray, config: dict,
                 device: torch.device | str, farm: str = "B",
                 ckpt_tag: str = "") -> dict[str, list[float]]:
    """train LSTMAE to reconstruct normal windows and return the loss history.
    train_windows is (N, W, F) float32, already MinMax-scaled."""

    cfg = config["lstmae"]
    device = torch.device(device)
    model.to(device)

    # el corte de validacion es la cola del array, cronologico y no aleatorio
    n = len(train_windows)
    n_val = max(1, int(n * cfg["val_fraction"]))
    n_train = n - n_val

    x_train = torch.tensor(train_windows[:n_train], dtype=torch.float32)
    x_val = torch.tensor(train_windows[n_train:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    # la reconstruccion se entrena con MSE, no con la L1 del AE del proyecto
    criterion = nn.MSELoss()
    es_cfg = cfg["early_stopping"]
    # el early stopping se queda con el mejor estado, no con el ultimo
    early_stop = EarlyStopping(
        patience=es_cfg["patience"],
        min_delta=es_cfg["min_delta"],
        restore_best_weights=es_cfg["restore_best_weights"],
    )

    models_dir = Path(config["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    logger.info(
        f"[LSTMAE] Farm {farm}: {n_train:,} train / {n_val:,} val windows "
        f"batch={cfg['batch_size']} max_epochs={cfg['max_epochs']}"
    )

    # una vuelta de entrenamiento y otra de validacion por epoca
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_losses.append(criterion(model(batch), batch).item())

        # la media de la epoca es lo unico que ve el early stopping
        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        logger.info(f"  [LSTMAE] epoch {epoch:>3}/{cfg['max_epochs']} train {t_loss:.6f} val {v_loss:.6f}")

        if early_stop.step(v_loss, model):
            logger.info(f"  [LSTMAE] early stop at epoch {epoch}, best val {early_stop.best_loss:.6f}")
            break

    # se guarda el mejor estado de la ejecucion, no el de la ultima epoca
    early_stop.restore(model)
    ckpt = models_dir / f"timeaddm_farm_{farm.lower()}{ckpt_tag}_lstmae.pt"
    torch.save(model.state_dict(), ckpt)
    logger.info(f"  [LSTMAE] saved to {ckpt}")

    return history


def train_diffusion(diffusion: nn.Module, lstmae: nn.Module, betas: torch.Tensor, train_windows: np.ndarray,
                    config: dict, device: torch.device | str, farm: str = "B", ckpt_tag: str = "") -> dict[str, list[float]]:
    """train the diffusion U-Net on frozen LSTMAE latents.
    train_windows is (N, W, F) float32 already MinMax-scaled, betas already on device."""

    cfg = config["diffusion"]
    device = torch.device(device)
    diffusion.to(device)
    lstmae.to(device)
    lstmae.eval()  # el LSTMAE queda congelado, aqui solo produce el latente

    # latent dataset, (N, W, H) permuted to (N, H, W) for Conv1d
    logger.info(f"[Diffusion] Farm {farm}: building latent dataset ...")
    all_latents: list[torch.Tensor] = []
    x_full = torch.tensor(train_windows, dtype=torch.float32)
    # lote x4, la codificacion no necesita gradiente
    loader_tmp = DataLoader(TensorDataset(x_full), batch_size=cfg["batch_size"] * 4, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader_tmp:
            z = lstmae.encode(batch.to(device))  # (b, W, H)
            all_latents.append(z.permute(0, 2, 1).cpu())  # (b, H, W)
    # el dataset latente entero se monta en CPU
    latents = torch.cat(all_latents, dim=0)

    # mismo corte de cola que en la fase 1, ahora sobre latentes
    n = len(latents)
    n_val = max(1, int(n * cfg["val_fraction"]))
    n_train = n - n_val

    z_train = latents[:n_train]
    z_val = latents[n_train:]

    train_loader = DataLoader(TensorDataset(z_train), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(z_val), batch_size=cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=cfg["learning_rate"])
    n_steps = betas.shape[0]  # T sale de la longitud del schedule
    es_cfg = cfg["early_stopping"]
    # mismo early stopping que la fase 1, ahora sobre la perdida de ruido
    early_stop = EarlyStopping(
        patience=es_cfg["patience"],
        min_delta=es_cfg["min_delta"],
        restore_best_weights=es_cfg["restore_best_weights"],
    )

    models_dir = Path(config["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / f"timeaddm_farm_{farm.lower()}{ckpt_tag}_diffusion.pt"
    # best weights saved every checkpoint_every epochs, an interrupted run still leaves a
    # usable diffusion on disk, no per-epoch resume
    checkpoint_every = int(cfg.get("checkpoint_every", 10))
    # with decay > 0, validation, early stop and checkpoint use the EMA weights, which
    # generalise better than the raw SGD iterate, decay <= 0 keeps the plain path
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = EMA(diffusion, ema_decay) if ema_decay > 0 else None
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    logger.info(
        f"[Diffusion] Farm {farm}: {n_train:,} train / {n_val:,} val latents "
        f"batch={cfg['batch_size']} max_epochs={cfg['max_epochs']} T={n_steps} "
        f"ema_decay={ema_decay if ema else 'off'}"
    )

    # por epoca, entrenamiento, validacion con los pesos EMA y checkpoint periodico
    for epoch in range(1, cfg["max_epochs"] + 1):
        diffusion.train()
        train_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            b_size = batch.size(0)
            # DDPM, un paso de ruido al azar y ruido gaussiano por muestra
            t = torch.randint(0, n_steps, (b_size,), device=device)
            e = torch.randn_like(batch)
            # la red predice el ruido inyectado y la perdida es su error contra e
            _, loss = noise_estimation_loss(diffusion, batch, t, e, betas)
            optimizer.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(diffusion.parameters(), cfg["grad_clip"])
            optimizer.step()
            if ema is not None:
                ema.update(diffusion)  # con los pesos ya movidos por el paso
            train_losses.append(loss.item())

        # validate on the EMA weights when EMA is active
        if ema is not None:
            ema.apply_shadow(diffusion)
        diffusion.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                b_size = batch.size(0)
                # la validacion tambien sortea t y ruido, la perdida lleva varianza
                t = torch.randint(0, n_steps, (b_size,), device=device)
                e = torch.randn_like(batch)
                _, loss = noise_estimation_loss(diffusion, batch, t, e, betas)
                val_losses.append(loss.item())

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        logger.info(f"  [Diffusion] epoch {epoch:>3}/{cfg['max_epochs']} train {t_loss:.6f} val {v_loss:.6f}")

        # captures the EMA weights, held by the model
        stop = early_stop.step(v_loss, diffusion)
        if checkpoint_every and epoch % checkpoint_every == 0 and early_stop.best_state is not None:
            torch.save(early_stop.best_state, ckpt)
            logger.info(f"  [Diffusion] checkpoint @ epoch {epoch} (best val {early_stop.best_loss:.6f}) -> {ckpt}")
        if ema is not None:
            ema.restore(diffusion)  # back to raw weights for the next epoch
        if stop:
            logger.info(f"  [Diffusion] early stop at epoch {epoch}, best val {early_stop.best_loss:.6f}")
            break

    # el mejor estado ya lleva los pesos EMA si estaba activa
    early_stop.restore(diffusion)
    torch.save(diffusion.state_dict(), ckpt)
    logger.info(f"  [Diffusion] saved to {ckpt}")

    return history
