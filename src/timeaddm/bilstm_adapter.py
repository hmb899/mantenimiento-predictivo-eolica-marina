"""
adapter that exposes a trained, frozen BiLSTMAutoencoder through TimeADDM's encode/decode
interface, so the diffusion U-Net can model the BiLSTM-AE's per-timestep bottleneck latents

the vanilla TimeADDM LSTMAE broadcasts a single bottleneck vector across all W timesteps. the
BiLSTM-AE uses return_sequences throughout, so its bottleneck is a genuine (batch, W, H)
sequence with temporal structure, which is exactly the (channels=H, length=W) tensor the
diffusion Conv1d U-Net wants, with no broadcasting hack

H, the diffusion's channel count, is encoder_units[-1] * (2 if bidirectional else 1). for
Farm C ([64, 16], bidirectional) → H = 32

the wrapped AE is frozen, only the diffusion trains on top of its latents, so this tests
directly whether the diffusion adds anomaly signal over the winning BiLSTM-AE's latent
manifold (CARE 0.7432)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.bilstm_autoencoder import BiLSTMAutoencoder


class BiLSTMAEAdapter(nn.Module):
    """frozen BiLSTMAutoencoder presented with .encode(x) giving (b, W, H) and .decode(z)
    giving (b, W, K)."""

    def __init__(self, ae: BiLSTMAutoencoder, freeze: bool = True) -> None:
        super().__init__()
        self.ae = ae
        # la ultima capa del encoder fija H, y una bidireccional concatena los dos sentidos
        last = self.ae.encoder_lstms[-1]
        directions = 2 if last.bidirectional else 1
        self.hidden_size: int = last.hidden_size * directions  # diffusion channel count H
        self.input_size: int = self.ae.encoder_lstms[0].input_size  # K (feature/component dim)
        if freeze:
            # sin gradiente en el AE, solo entrena la difusion que va encima
            for p in self.ae.parameters():
                p.requires_grad_(False)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """run the encoder LSTMs and return the bottleneck sequence (batch, W, H), no
        dropout."""

        # se encadenan las capas del encoder a mano, saltandose el forward del AE
        out = x
        for lstm in self.ae.encoder_lstms:
            out, _ = lstm(out)
        return out


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """run the decoder LSTMs and the output projection, returning the reconstruction
        (batch, W, K)."""

        out = z
        for lstm in self.ae.decoder_lstms:
            out, _ = lstm(out)
        return self.ae.output_layer(out)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
