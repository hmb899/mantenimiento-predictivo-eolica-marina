"""C1 — Earliness-first por desviación de trayectoria (Modelo C, veta B).

En vez de RECONSTRUIR la ventana (el AE interpola el drift -> avisa tarde), PRONOSTICA la trayectoria
latente normal (componentes PCA del camino del AE) a horizonte multi-paso y puntúa la DESVIACIÓN
respecto al pronóstico. Un fallo incipiente rompe la trayectoria esperada antes de que el nivel esté
lejos del manifold -> ataca la Earliness (cuello de Farm C) por diseno. Refs: precedentes 2026 de
forecast-departure/desacuerdo (FATE arXiv:2602.17028, RED-F arXiv:2511.20044). Kill-gate ANTES de
multi-seed: corr-AE < 0.5 Y CARE-CV solo >= 0.68 (seed-42).
"""
