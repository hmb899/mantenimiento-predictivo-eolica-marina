# `src/`

## Portado o reimplementado de la literatura

| Paquete | Qué implementa | Procedencia |
|---|---|---|
| `timeaddm/` | Difusión multinivel sobre latentes recurrentes | Portado de la implementación oficial (Hu et al., ICASSP 2024) con adaptaciones. La U-Net sigue a Ho et al. (2020) |
| `imdiffusion/` | Detección por imputación con máscaras de rejilla complementarias | Reimplementado de Chen et al. (VLDB 2023); lo resuelto contra el código original va marcado `# ref: IMDiffusion` |
| `gdn/` | Grafo aprendido por *embeddings* y pronóstico por atención | Reimplementado de Deng & Hooi (AAAI 2021), en PyTorch plano y sin PyTorch Geometric |
| `crgn/` | Dos codificadores GCN residuales entrenados con InfoNCE | Reimplementado de Yang, Zhang & Yu (Energies 2025), y solo el núcleo: sin AFS, sin *health index* ni umbral SPC |
| `driftdiff/` | D3R: descomposición dinámica con reconstrucción por difusión | Portado de ForestsKing/D3R (NeurIPS 2023) |

`imdiffusion/` y `gdn/` se reimplementaron en vez de importarse porque el repositorio oficial del artículo no lleva licencia. De `crgn/` se reimplementó
solo el núcleo del método. `timeaddm/` y `driftdiff/` sí parten de la implementación oficial de sus
autores.

## Mixtos

| Paquete | Qué implementa | Procedencia |
|---|---|---|
| `graphdiff/` | `model.py` y `scorer.py`, difusión latente sobre grafo. Y `physics.py`, el *score* físico de coherencia | GraphDiff es una adaptación, no un port literal, de TSAD-C (UAI 2025) y NGLS-Diff (ECML-PKDD 2024). `physics.py` es propio |
| `models/` | El autoencoder BiLSTM desplegado, su entrenador, el CUSUM y el evaluador CARE del proyecto | La arquitectura del autoencoder es de Barnabei et al. (2024); el CUSUM y el evaluador son propios |

## Desarrollo propio

| Paquete | Qué implementa | Nota                                                                                                                                                                                         |
|---|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `care_official/` | La métrica CARE del *benchmark*, con la que se mide todo lo demás | Implementación propia de la métrica definida por Guck et al. (2024). Se escribió y se validó en un repositorio auxiliar aparte, con 82 pruebas unitarias; aquí solo se adaptaron los imports |
| `preprocessing/` | Las diez etapas del *pipeline* en su único orden válido |  el escalado pertenece al *pipeline* del modelo (Barnabei et al., 2024)                                                                                                                      |
| `agents/` | Los dos agentes, los contratos, el cliente Ollama y cuatro herramientas | Las reglas calibradas deciden; el LLM solo redacta                                                                                                                                           |
| `scheduler/` | El problema multiobjetivo, el frente, el *greedy*, la comprobación exacta, ERA5 y el catálogo | El motor NSGA-II es pymoo; la formulación de los tres objetivos es del trabajo                                                                                                               |
| `model_prep/` | Escalado, PCA, limpieza por Isolation Forest y la reducción por filas | Sobre scikit-learn                                                                                                                                                                           |
| `xai/` | Cargadores verificados, atribución en tres niveles, CUSUM y figuras | Ninguna figura se construye sobre un cargador que no haya pasado `verify_xai_loaders.py`                                                                                                     |
| `envmod/` | El modelo acústico de la emisión operacional y la base de especies | Lo que no procede de una fuente citable va marcado `assumption`                                                                                                                              |
| `trajdev/` | Pronóstico de la trayectoria normal y desviación respecto a él | El mejor detector individual del trabajo                                                                                                                                                     |
| `fusion/` | Normalizadores y combinadores de *score*, todos puros | La cola gaussiana es de Kriegel et al. (2011); los combinadores, de Aggarwal & Sathe (2017)                                                                                                  |

