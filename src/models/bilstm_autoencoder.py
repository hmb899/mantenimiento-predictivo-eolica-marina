"""
autoencoder LSTM bidireccional para deteccion de anomalias en aerogeneradores

BiLSTMAutoencoder
    arquitectura de Barnabei et al. (2024), el decoder es espejo del encoder
    entrada (batch, W, K), las capas las fija encoder_units, todas con return_sequences=True
    Barnabei encadena BiLSTM(128) y BiLSTM(64), y el proyecto pasa [64, 16]
    salida TimeDistributed(Linear(K) → ReLU) → (batch, W, K)
    reconstruye cada ventana y el score de anomalia es el MAE contra la entrada
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMAutoencoder(nn.Module):
    """autoencoder BiLSTM que detecta anomalias por error de reconstruccion."""

    def __init__(self, input_size: int, encoder_units: list[int] = None, decoder_units: list[int] = None,
                 dropout: float = 0.0, bidirectional: bool = True, out_activation: str = "relu") -> None:
        super().__init__()
        # ReLU solo vale con objetivos no negativos (MinMax) y con StandardScaler la salida
        # debe ser lineal; Identity deja iguales las claves "output_layer.0.*" del state_dict
        self.out_activation = out_activation
        if encoder_units is None:
            encoder_units = [128, 64]
        if decoder_units is None:
            decoder_units = [64, 128]

        directions = 2 if bidirectional else 1  # cada BiLSTM concatena las dos direcciones

        # el encoder encadena capas, la salida de una es la entrada de la siguiente
        # el dropout se salta en la ultima, cuya salida ya es el latente
        self.encoder_lstms: nn.ModuleList = nn.ModuleList()
        self.encoder_drops: nn.ModuleList = nn.ModuleList()
        in_size = int(input_size)
        for i, units in enumerate(encoder_units):
            self.encoder_lstms.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            if dropout > 0 and i < len(encoder_units) - 1:
                self.encoder_drops.append(nn.Dropout(dropout))
            in_size = units * directions

        # el decoder es espejo y arranca del latente, ya concatenado en las dos direcciones
        self.decoder_lstms: nn.ModuleList = nn.ModuleList()
        self.decoder_drops: nn.ModuleList = nn.ModuleList()
        in_size = encoder_units[-1] * directions
        for i, units in enumerate(decoder_units):
            self.decoder_lstms.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=units,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            if dropout > 0 and i < len(decoder_units) - 1:
                self.decoder_drops.append(nn.Dropout(dropout))
            in_size = units * directions

        self._dropout = dropout
        # la capa de salida proyecta de vuelta a input_size en cada paso de tiempo
        _act = nn.ReLU() if out_activation == "relu" else nn.Identity()
        self.output_layer = nn.Sequential(
            nn.Linear(decoder_units[-1] * directions, input_size),
            _act,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pasa por el encoder y por el decoder y proyecta de vuelta al espacio de entrada."""

        out = x
        # encoder
        for i, lstm in enumerate(self.encoder_lstms):
            out, _ = lstm(out)
            # encoder_drops[i] solo existe si i no es la ultima capa, el guard lo garantiza
            if self._dropout > 0 and i < len(self.encoder_lstms) - 1:
                out = self.encoder_drops[i](out)

        # decoder
        for i, lstm in enumerate(self.decoder_lstms):
            out, _ = lstm(out)
            if self._dropout > 0 and i < len(self.decoder_lstms) - 1:
                out = self.decoder_drops[i](out)

        return self.output_layer(out)


    def reconstruction_error(self, x: torch.Tensor, reduction: str = "mean",
                             mode: str = "mae_window") -> torch.Tensor:
        """error de reconstruccion por muestra, mae_window sobre la ventana entera por
        defecto o l2_last solo en el ultimo paso, los dos leen la misma reconstruccion."""

        recon = self(x)
        if mode == "l2_last":
            # readout de PruebasTFM (BiLSTM-NBM), norma L2 del error en el ultimo paso
            # post-hoc sobre la misma reconstruccion, sin reentrenar, ignora reduction
            d = x[:, -1, :] - recon[:, -1, :]
            return d.pow(2).sum(dim=1).sqrt()
        err = torch.abs(x - recon)
        # mae_window promedia sobre todo el bloque de pasos × canales, un escalar por ventana
        if reduction == "mean":
            return err.mean(dim=(1, 2))
        # sin reduccion sale el error crudo con la forma de la entrada
        return err
