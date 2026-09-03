"""
fusion a nivel de score

combina los scores de anomalia por ventana de detectores ya entrenados (AE, CRGN, difusion,
GDN...) sin reentrenar nada. `combiners` son los normalizadores y las funciones de
combinacion, todas puras.
"""
