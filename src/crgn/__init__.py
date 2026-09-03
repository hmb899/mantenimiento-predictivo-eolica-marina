"""
nucleo CRGN, Contrastive Residual Graph Network (Yang, Zhang & Yu, Energies 2025)

segundo modelo de grafos del proyecto, hermano de src/gdn. cada variable es un nodo, el grafo
Spearman es fijo por parque y dos encoders GCN residuales con pesos compartidos se entrenan
con InfoNCE auto-supervisado sobre operacion normal. el score por ventana es la desviacion del
embedding respecto a la distribucion de los normales, lo pone CRGNScorer, y el umbral y el
evento los pone el harness.

es el nucleo CRGN y no el pipeline AFS-CRGN, sin AFS de multiplicadora, health index ni
umbral SPC.
"""
