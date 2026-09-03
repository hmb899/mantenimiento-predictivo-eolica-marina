"""
LSTM autoencoder, the recurrent temporal embedding stage of TimeADDM

references Hu et al. (ICASSP 2024), TIMEEMB component
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAE(nn.Module):
    """LSTM autoencoder that compresses multivariate windows into a latent sequence. the
    encoder takes (batch, seq_len, input_size) and gives the full output sequence (batch,
    seq_len, hidden_size), preserving temporal structure, and the decoder rebuilds (batch,
    seq_len, input_size) from the latent."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0,
                 linear_output: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # linear_output=True drops the input ReLU and uses an identity output, so the AE takes
        # StandardScaler-style signed unbounded inputs, the BiLSTM-AE's winning lever
        # False keeps the published TimeADDM design, input ReLU and Sigmoid, for MinMax [0,1]
        self.linear_output = linear_output

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.output_act = nn.Identity() if linear_output else nn.Sigmoid()


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """map (batch, seq_len, input_size) to (batch, seq_len, hidden_size)."""

        if not self.linear_output:
            x = torch.relu(x)
        _, (h, _) = self.encoder_lstm(x)
        # el LSTM devuelve un unico estado final, y aqui se difunde a los seq_len pasos, asi
        # que el latente no tiene estructura temporal propia (lo que corrige BiLSTMAEAdapter)
        return h.squeeze(0).unsqueeze(1).expand(-1, x.size(1), -1)


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """map (batch, seq_len, hidden_size) back to (batch, seq_len, input_size)."""

        out, _ = self.decoder_lstm(z)
        return self.output_act(self.output_layer(out))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode then decode, reconstruction in [0, 1] with Sigmoid or unbounded if
        linear_output."""

        return self.decode(self.encode(x))
