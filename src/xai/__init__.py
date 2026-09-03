"""
infraestructura compartida de los 4 notebooks de explicabilidad

ninguna figura se construye sobre un cargador que no haya pasado
scripts/verify_xai_loaders.py, que reproduce exacto los dumps y los gates committeados.
"""
from src.xai.loaders import (  # noqa: F401
    DEVICE,
    load_ae,
    load_ae_events,
    load_meta,
    load_scores,
    load_trajdev,
    load_trajdev_events,
    trajdev_channel_z,
)
from src.xai.cusum import cusum_series, first_alarm, h_sweep_best, official_cusum  # noqa: F401
