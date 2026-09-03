"""
bucle de entrenamiento del BiLSTMAutoencoder con early stopping y checkpoints

EarlyStopping
    para segun la perdida de validacion y guarda el mejor estado visto
Trainer
    particion temporal train/val, bucle de epocas, checkpoint periodico y guardado
    del mejor modelo
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """para el entrenamiento cuando la perdida de validacion deja de mejorar."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, restore_best_weights: bool = True) -> None:
        """inicializa los contadores y el seguimiento del mejor estado."""

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best_weights
        self.best_loss: float = float("inf")
        self.best_state: dict | None = None
        # cuenta epocas consecutivas sin mejora, no epocas totales
        self.counter: int = 0


    def step(self, val_loss: float, model: nn.Module) -> bool:
        """devuelve True si el entrenamiento debe parar por la mejora de la perdida
        de validacion."""

        # solo cuenta como mejora si baja de best_loss por mas de min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                # la copia va a CPU para no retener memoria de la GPU durante el resto del
                # entreno
                self.best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


    def restore(self, model: nn.Module) -> None:
        """devuelve al modelo los mejores pesos guardados."""

        # sin restore_best, o si nunca hubo mejora, el modelo se queda como esta
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    """entrena un BiLSTMAutoencoder con early stopping y checkpoints."""

    def __init__(self, model: nn.Module, config: dict, device: torch.device | str = "cpu") -> None:
        """prepara el optimizador, el criterion, el early stopping y la config de
        checkpoint."""

        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

        ae_cfg = config["autoencoder"]
        lr = ae_cfg["learning_rate"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # el error de entrenamiento es el mismo que luego puntua el AnomalyDetector
        self.criterion = nn.L1Loss()  # MAE

        self.max_epochs: int = ae_cfg["max_epochs"]
        self.batch_size: int = ae_cfg["batch_size"]
        self.val_fraction: float = ae_cfg["val_fraction"]

        es_cfg = ae_cfg["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
            restore_best_weights=es_cfg["restore_best_weights"],
        )

        chk_cfg = config.get("output", {})
        self.models_dir = Path(chk_cfg.get("models_dir", "results/models"))
        self.checkpoint_every = chk_cfg.get("checkpoint_every_n_epochs", 10)
        # results/models/ guarda artefactos congelados que ningun nivel del gate vigila
        # sobrescribir uno exige --overwrite-frozen explicito
        self.save_best_model = chk_cfg.get("save_best_model", True)
        self.overwrite_frozen = chk_cfg.get("overwrite_frozen", False)

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}


    def fit(self, train_windows: np.ndarray, farm: str = "B") -> dict[str, list[float]]:
        """entrena el autoencoder y devuelve el historial de perdidas."""

        n = len(train_windows)
        # al menos una ventana de validacion aunque el train sea diminuto
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val

        # la particion temporal conserva el orden, la ultima val_fraction es la validacion
        x_train = torch.tensor(train_windows[:n_train], dtype=torch.float32)
        x_val = torch.tensor(train_windows[n_train:], dtype=torch.float32)

        # el train se baraja por lote, la validacion se recorre siempre igual
        train_loader = DataLoader(
            TensorDataset(x_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(x_val),
            batch_size=self.batch_size,
            shuffle=False,
        )

        logger.info(
            f"[Trainer] Farm {farm} "
            f"train {n_train:,} / val {n_val:,} windows "
            f"batch {self.batch_size} max {self.max_epochs} epochs"
        )

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # por epoca, una vuelta de entrenamiento, otra de validacion y checkpoint periodico
        for epoch in range(1, self.max_epochs + 1):
            # entrenamiento
            self.model.train()
            train_losses: list[float] = []
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch)
                # el autoencoder se entrena contra su propia entrada, no hay etiquetas
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            # validacion
            # eval y sin gradiente, asi el dropout no interviene en la perdida de validacion
            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    recon = self.model(batch)
                    loss = self.criterion(recon, batch)
                    val_losses.append(loss.item())

            # la media de la epoca es lo unico que ven el historial y el early stopping
            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            logger.info(
                f"  Epoch {epoch:>3}/{self.max_epochs} "
                f"train_loss {train_loss:.6f}  val_loss {val_loss:.6f}"
            )

            # checkpoint
            # una ejecucion interrumpida deja en disco los pesos de la ultima epoca multiplo
            if epoch % self.checkpoint_every == 0:
                ckpt_path = self.models_dir / f"farm_{farm.lower()}_epoch{epoch:03d}.pt"
                torch.save(self.model.state_dict(), ckpt_path)

            # early stopping
            # step guarda por dentro el mejor estado y devuelve True al agotarse la paciencia
            if self.early_stopping.step(val_loss, self.model):
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"best val_loss {self.early_stopping.best_loss:.6f}"
                )
                break

        # se recupera el mejor estado de la ejecucion, no el de la ultima epoca
        self.early_stopping.restore(self.model)

        # se guarda el mejor modelo final
        # tres caminos, no escribir nada, conservar el congelado o sobrescribirlo
        best_path = self.models_dir / f"farm_{farm.lower()}_best.pt"
        if not self.save_best_model:
            logger.info(f"  save_best_model=false: no se escribe {best_path}")
        elif best_path.exists() and not self.overwrite_frozen:
            logger.warning(f"{best_path} ya existe: se CONSERVA. Es un artefacto congelado; "
                           f"pasa --overwrite-frozen si de verdad quieres sobrescribirlo.")
        else:
            torch.save(self.model.state_dict(), best_path)
            logger.info(f"  Best model saved to {best_path}")

        return self.history
