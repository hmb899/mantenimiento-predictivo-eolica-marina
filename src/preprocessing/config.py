"""carga la configuracion de preproceso desde un YAML"""
from pathlib import Path
import yaml


def load_config(path: str | Path | None = None) -> dict:
    """carga el YAML de configuracion, config/preprocessing_config.yaml por defecto."""

    if path is None:
        # tres niveles sobre src/preprocessing/ esta la raiz del repositorio
        path = (Path(__file__).resolve().parent.parent.parent / "config" / "preprocessing_config.yaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
