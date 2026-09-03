"""GraphDiff (Track B, Modelo B): difusion latente condicionada por grafo sobre el encoder CRGN.

Adaptacion (NO port literal) de TSAD-C (UAI 2025, arXiv:2308.12563) x NGLS-Diff (ECML-PKDD 2024,
doi:10.1007/978-3-031-70352-2_17), motivada por los hallazgos del proyecto:
  - encoder de grafo = ResidualGCN de src/crgn (reuso) sobre la representacion top-100 de CRGN;
  - latente normal-gathering (VAE + KL a N(0,I)), estilo NGLS-Diff, para que la difusion discrimine;
  - difusion latente (U-Net 1D de src/timeaddm, reuso) sobre el eje de nodos;
  - SCORE PRIMARIO = eps nativo ||e - eps_theta(z_t, L)||^2 (sin decodificar -> candidato ortogonal),
    con recon como hedge. Ojo: los papers puntuan por recon (redundante con el AE aqui); la apuesta
    del Modelo B es que el eps sobre el latente de GRAFO normal-gathering sea fuerte Y ortogonal.

Codigo nuevo aislado. Reuso SOLO LECTURA de src/{crgn,timeaddm,care_official} y run_gdn/run_model.
"""
