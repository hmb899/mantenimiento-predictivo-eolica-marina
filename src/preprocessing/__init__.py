"""
pipeline de preproceso del SCADA de aerogeneradores offshore de CARE

para empezar

    from src.preprocessing import PreprocessingPipeline
    from src.preprocessing.config import load_config

    config = load_config()
    pipeline = PreprocessingPipeline(config)

    result_b = pipeline.run("B")   # Farm B
    result_c = pipeline.run("C")   # Farm C
"""
from .config import load_config
from .pipeline import PreprocessingPipeline, PipelineResult
from .windower import TestSequence, TestSequenceDict

__all__ = [
    "load_config",
    "PreprocessingPipeline",
    "PipelineResult",
    "TestSequence",
    "TestSequenceDict",
]
