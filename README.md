# Mantenimiento Predictivo para Aerogeneradores Offshore con Restricciones Ambientales


La operación y el mantenimiento de los parques eólicos marinos concentran entre el 25% y el 50%
del coste del ciclo de vida de la instalación, y sus intervenciones comparten el mar con especies
protegidas cuyos períodos sensibles coinciden con los meses de mejor acceso. Este trabajo propone
una metodología de mantenimiento predictivo que incorpora esa restricción ambiental en la
planificación, aplicada a aerogeneradores marinos y construida sobre los dos parques marinos del
*benchmark* público CARE. Sobre un marco de evaluación riguroso (métrica oficial verificada,
validación cruzada por eventos y protocolo *multiseed* con comparación semilla a semilla) se
comparan cinco familias de detección de anomalías: autoencoders, difusión, grafos predictivos y
contrastivos, coherencia física y pronóstico de trayectoria.

Dos de esas familias son desarrollos propios, el modelo de desviación de trayectoria, que resulta
el mejor detector individual, y el *score* físico de coherencia, cuya fusión con el autoencoder y
la red contrastiva alcanza, en el mayor de los dos parques, una puntuación CARE de 0.7694 ± 0.007
bajo validación cruzada, frente al 0.6364 del *baseline* reproducido del artículo del *benchmark*.
Sobre las alarmas operan una capa de explicabilidad verificable, agentes de diagnóstico y
categorización que trabajan sin acceso a las etiquetas y un planificador multiobjetivo que programa
las intervenciones con meteorología real y una base de conocimiento ambiental citable, todo ello
integrado en una consola de operador. En el marco de costes esperados del planificador, el sistema
ahorra el 40% de la factura reactiva de los fallos que captura. Respetar el calendario de la fauna
cuesta el 28% de lo que ahorraría un planificador de mínimo coste y compra a cambio un 41% menos
de perturbación sobre las especies, de modo que el criterio ambiental deja de ser una posición de
principio y pasa a tener un precio y un beneficio estimados.

---

## Qué hay en el repositorio

El sistema recorre cuatro pasos, y el repositorio se ordena por ellos:

```
Datos SCADA  →  Detección de anomalías  →  Diagnóstico y evidencia  →  Planificación ambiental
registros de     alarma con semanas         canal responsable y         coste, riesgo e impacto
10 min de dos    de antelación              urgencia estimada           sobre la fauna
parques marinos
```

Los cuatro objetivos específicos del trabajo se corresponden uno a uno con esos pasos:

| Objetivo | Dónde está su código |
|---|---|
| Establecer un marco de evaluación riguroso sobre el *benchmark* CARE | `src/care_official/`, `src/models/evaluator.py`, `scripts/run_model.py::evaluate` |
| Desarrollar modelos de detección de anomalías | `src/{models,timeaddm,imdiffusion,gdn,crgn,trajdev,graphdiff,driftdiff,fusion}/` |
| Dotar al sistema de explicabilidad verificable y de agentes de diagnóstico | `src/xai/`, `src/agents/` |
| Formular la planificación multiobjetivo del mantenimiento con criterio ambiental | `src/scheduler/`, `src/envmod/`, `webapp/` |

El resto del árbol:

| Ruta | Contenido |
|---|---|
| `scripts/` | Los ejecutables: cinco librerías compartidas, quince *runners* de detector, tres gates y seis ficheros en cuatro subdirectorios |
| `notebooks/` | Ocho cuadernos con sus salidas incrustadas, que recorren el trabajo del análisis exploratorio al planificador |
| `results/` | Los artefactos numéricos y los pesos congelados, en tres subdirectorios |
| `config/` | La configuración del preprocesado y de cada familia de modelos |
| `webapp/` | La consola de operador|
| `tests/` | 149 pruebas, todas sobre `src/` |


El dataset no viaja con el código y `data/` contiene solo un marcador vacío. Tampoco viajan las
series de *score* por ventana que consume la capa de fusión, 2,7 GiB, ni las cachés de
representación, ni las figuras. Es lo que decide qué se puede ejecutar aquí y qué no.

---

## El entorno

Requiere Python 3.12 o superior.

**En Windows, descomprime el repositorio en una ruta corta**, del tipo `C:\tfm`. Sin
`LongPathsEnabled` activado sigue vigente el límite de 260 caracteres, que alcanza a las
extensiones compiladas de `scikit-learn` dentro de `.venv`: entonces `pytest` aborta en la
recolección, sin ejecutar una sola prueba, con un `ModuleNotFoundError` de
`sklearn.metrics._pairwise_distances_reduction._middle_term_computer` que no apunta a la causa.

```bash
uv sync
```

Instala las versiones exactas de `uv.lock`. La pila de la que dependen las cifras es la misma
del entorno con el que se producen los resultados: torch 2.10.0+cu128, numpy, scipy,
scikit-learn y pandas. Lo que separa a los dos son ocho dependencias de borde.

Dos grupos opcionales, que no se instalan por defecto:

```bash
uv sync --extra dev        # para la batería de pruebas
uv sync --extra webapp     # para la consola de operador
```

Existe también `pip install -e .`, que no requiere `uv`. Resuelve desde `pyproject.toml`, cuyas
cotas son rangos `>=`, así que puede dar versiones más nuevas que las del trabajo.

Los comandos de este documento asumen el intérprete del entorno del proyecto, porque `python` a
secas puede resolver a otra instalación del sistema. Con `uv` basta anteponer `uv run`; con `pip`,
activar el entorno o invocarlo de forma explícita, `.venv/Scripts/python.exe` en Windows y
`.venv/bin/python` en Linux y macOS.

El entrenamiento y la puntuación de los modelos necesitan una GPU con CUDA. El planificador y los
agentes no importan `torch` en ningún punto y funcionan en CPU.

---

## Qué se puede ejecutar
Hay que distinguir entre lo que funciona sobre este repositorio tal cual, lo que necesita el dataset CARE y una
GPU, y lo que no es reproducible aquí en ningún caso.

| | Dataset | GPU | Ollama | Coste |
|---|:-:|:-:|:-:|--:|
| `pytest -m "not slow"` | no | no | no | ~9 s |
| `scripts/scheduler_smoke.py` | no | no | no | ~45 s |
| `scripts/agents_smoke.py` | no | no | opcional | ~30 s |
| Leer los ocho cuadernos | no | no | no | — |
| Consola de operador | sí | sí | opcional | ~30–45 min de preparación |
| Un detector en su mejor configuración | sí | sí, salvo la física | no | ~25 min – ~10 h |
| Reconstruir el corpus preprocesado | sí | no | no | ~30 min |
| La capa de fusión | sí | sí | no | ~3 h, con sus tres detectores antes |

### Sobre el repositorio, sin nada más

Tres comandos se ejecutan sobre un clon sin nada más: sin dataset, sin GPU y sin credenciales
externas.

```bash
python -m pytest -m "not slow"        # 149 pruebas, una marcada lenta que se salta sola
python scripts/scheduler_smoke.py     # planificador multiobjetivo (NSGA-II)
python scripts/agents_smoke.py        # contratos, módulo ambiental y los dos agentes
```

El del planificador hace algo más que arrancar: enumera el frente exacto por fuerza bruta y
comprueba que el que devuelve NSGA-II lo alcanza con holgura normalizada 0.0000. Es la única
cobertura automática que tiene el capítulo de planificación.

El de los agentes cierra con un sondeo a un servidor Ollama local, un servicio externo y opcional
que no viaja con el código. Sin él en marcha el script termina con
`SMOKE F1-F2-F5: FALLOS ['ollama']` y código 1, aunque los veinticuatro comprobantes anteriores
hayan pasado. Que falte no cambia ninguna decisión del sistema: deciden las reglas calibradas y el LLM solo reescribe 
en prosa lo que ya está redactado. 

### Los ocho cuadernos, que se leen

Vienen con sus salidas incrustadas y se recorren de principio a fin sin ejecutar nada. Para la
mayoría de las cifras que publica la memoria, ese *output* embebido es el único respaldo versionado
que existe en el repositorio, de modo que volver a correrlos sustituiría la evidencia por una
medición nueva.

| Cuaderno | Qué muestra |
|---|---|
| `00_EDA.ipynb` | El *benchmark* sin preprocesar: volumen, calidad temporal, catálogo de sensores y estructura de los eventos |
| `00_preprocessing.ipynb` | El *pipeline* etapa por etapa sobre los dos parques, con el recuento de fila de entrada y de salida de cada una |
| `01_trajdev_farm_c.ipynb` | Explicabilidad del mejor detector individual antes del fallo: oclusión, gradientes integrados y contrafactual |
| `02_ensemble_campeon_farm_c.ipynb` | El reparto de la detección entre los tres miembros del *ensemble* campeón, con su Shapley |
| `03_trajdev_farm_b.ipynb` | La réplica del modelo propio en el parque B, más ruidoso, y lo que cambia el multi-semilla |
| `04_ensemble_farm_b.ipynb` | El mismo *ensemble* en el parque B, donde la mejora viene de la varianza y no de la cobertura |
| `05_diagnostico_agente1.ipynb` | El triaje del agente de diagnóstico, medido sin concesiones sobre los quince eventos del parque B |
| `06_planificador_agente2.ipynb` | Del diagnóstico al calendario: coste, riesgo y fauna, con la tabla del ahorro |

### Con el dataset CARE y una GPU

El dataset ocupa 20 GB, de los cuales 19 corresponden a los parques B y C. No es redistribuible y
debe descargarse aparte en `data/raw/CARE_To_Compare/`.

**La consola de operador.** Muestra el sistema completo sobre un evento real, de la alarma a la
decisión. Necesita cuatro pasos de preparación, porque las series de *score* y la caché de
representación sobre las que trabaja no viajan con el código:

```bash
python scripts/signal_atlas.py --rebuild --cache-tag 3 --with-val --build-only          # ~22 min, CPU
python scripts/run_trajdev_c.py --seeds 42 --repr atlas --atlas-tag 3 --n-models 3 \
       --npz-stem trajdev3 --dump-scores                                                #  ~6 min, GPU
python scripts/run_model.py --farm B --dump-scores                                      # ~13 min, GPU
python scripts/run_trajdev_b.py --seeds 42 --dump-scores                                #  ~2 min, GPU

python webapp/desktop.py                          # ventana nativa
uvicorn webapp.api.main:app                       # navegador, http://127.0.0.1:8000
```

Los dos primeros pasos bastan para el parque C, que es el que la consola abre por defecto, y son
unos 30 minutos; los cuatro, unos 45. Generan 6,0 GiB de caché de atlas y 0,7 GiB de series de
*score*. Sin ellos el servidor arranca y la interfaz se dibuja, pero cualquier petición que toque
datos responde con un error 500.

Cuatro detalles de esos comandos son fáciles de pasar por alto:

- `--with-val` añade los canales sobre los que puntúa el detector que usa la consola, así que
  no es opcional.
- `--n-models` vale 3 por defecto en el parque B y 1 en el C: en C hay que pasarlo.
- `--build-only` termina antes de escribir `results/tables/signal_atlas_c.json`, y sin él lo sobrescribe.
- El tercer paso está por el corpus y no por el *score*. La consola no usa el autoencoder del parque
  B, pero sí los dos ficheros de secuencias de test que ese comando deja de paso.

La meteorología ERA5 y el terminal LLM son opcionales y ninguna decisión del sistema depende de
ellos: sin ERA5 el escenario usa meteorología sintética, y sin Ollama solo falla el terminal.

**Los detectores, cada uno en su mejor configuración.** Los *runners* aceptan muchos modos y ninguno
marca cuál produjo la cifra de la memoria, así que esta es la correspondencia. El coste es el de una
ejecución desde cero en el parque C, con la preparación de la representación incluida.

| Familia | Comando de la configuración publicada | Desde cero | Cifra |
|---|---|--:|---|
| AE BiLSTM | `run_model.py --farm C --dump-scores` · `--farm B --dump-scores` | ~55 min | C 0.7432 / B 0.7211 |
| TrajDev | `run_trajdev_c.py --seeds 42,7,123,1234,12345 --repr atlas --atlas-tag 3 --n-models 3 --npz-stem trajdev3 --dump-scores` | ~30 min | C 0.7399 ± 0.010 |
| CRGN | `run_crgn.py --farm C --dump-scores --seed 42` | ~25 min | C 0.7285 ± 0.003 |
| GDN | `run_gdn.py --farm C --seed 42` | ~25 min | C 0.7123 ± 0.027 |
| Física | `dump_phys_scores.py --farm C --seeds 42,7,123,1234,12345` | ~23 min | C 0.674 |
| TimeADDM | `run_timeaddm_bilstm_c.py --ae-mode frozen --seed 42` | ~90 min | C 0.7384 ± 0.004 |
| ImDiffusion | `run_imdiffusion.py --farm C --features top100 --mask-mode feature_offset --regime --split 3 --seed 42` | ~10 h | C 0.7095 ± 0.003 |

El coste del autoencoder está dominado por el volcado y no por el entrenamiento. Entrenarlo y
evaluarlo en el parque C son unos 9 minutos, sobre una representación que cuesta unos 21 en
construirse; volcar la serie de *score* que consume la fusión lleva la ejecución a unos 33, porque
son 157 020 ventanas a paso 1. Sin `--dump-scores` el total desde cero baja a unos 30 minutos.

El *score* físico no entrena nada, no necesita GPU y vuelca sus seis variantes de umbral y de
agregación en una sola pasada. En el otro extremo, una sola semilla de ImDiffusion en el parque C
cuesta más que las otras cuatro familias juntas, en los dos parques, con sus cinco semillas.

**La reconstrucción del corpus.** Cada parque tiene su propia configuración y son
distintas por diseño, porque el suavizado de la señal mejora los resultados en el parque B y los
empeora en el C:

```bash
python scripts/preproc/run_preprocessing.py --farm B --patch config/experiments/l6_light.yaml   --out mi_corpus
python scripts/preproc/run_preprocessing.py --farm C --patch config/experiments/canonico_C.yaml --out mi_corpus
```

El resultado son dos tensores de 411 249 × 27 × 30 y 2 358 533 × 27 × 35, idénticos bit a bit a los
del trabajo, en aproximadamente un minuto para el parque B y media hora para el C.

En el parque C ese corpus no es el último paso. El autoencoder desplegado, los cuadernos y la fusión
consumen otra representación, que se queda con las 100 variables promediadas más informativas de las
217 disponibles y las reduce a 55 componentes principales. La construye `run_model` en memoria
durante su propia ejecución, así que no aparece en el directorio de salida y su ausencia no indica que
falte nada.

### Lo que exige recorrer antes la cadena entera

Dos cosas no se pueden lanzar sobre un clon, y no porque falte código: consumen las series de
*score* por ventana, que son 2,7 GiB y no viajan, y el dataset crudo. Con el CARE descargado y los
modelos ejecutados antes, las dos funcionan.

**La capa de fusión.** `scripts/fuse_multi.py` consume una serie de *score* por detector y por
semilla, y antes de puntuar nada construye el evaluador oficial sobre `data/raw/CARE_To_Compare`.
Incluir esas series no ahorraría el paso, porque seguiría faltando el dataset. El camino es, en este
orden: descargar el CARE, ejecutar los modelos que producen las series y lanzar `fuse_multi.py`. El
resultado publicado del *ensemble* campeón sale de esta invocación:

```bash
python scripts/fuse_multi.py --farm C --detectors ae,crgn,phys95maxrate \
       --seeds 42,7,123,1234,12345 --normalizers zscore --tag <etiqueta>
```

**La verificación de los cargadores.** `scripts/verify_xai_loaders.py` comprueba que los cargadores
de `src/xai` reproducen los volcados congelados, y es el script sobre el que se levantan los cuatro
cuadernos de explicabilidad. Tampoco arranca sobre un clon: abre las series de *score*, las
secuencias de test del parque B y el evaluador oficial sobre el dataset crudo.

### Lo único que no se puede reproducir

**TimeADDM en el parque B.** Su autoencoder desplegado, `farm_b_best.pt`, se perdió durante el
desarrollo. Los pesos de los difusores de ese
parque sí se conservan, pero resultan inservibles sin el autoencoder cuyo espacio latente modelan.
Entrenar uno nuevo no resolvería nada: produciría un espacio latente distinto y, por tanto, un
detector distinto. Su cifra sigue siendo verificable en `results/tables/`; lo que no puede
repetirse es su cálculo exacto.

---

## Antes de ejecutar un modelo

Los *runners* escriben en `results/models/` y en `results/tables/`. Las dos rutas contienen
artefactos versionados que, una vez sobrescritos, el repositorio no siempre puede reponer.

**1 · Comprueba el árbol después de cada ejecución.**

```bash
git status --short results/models/ results/tables/     # debe salir '??', nunca 'M'
git checkout -- results/models/ results/tables/        # si sale alguna 'M'
rm -f results/models/farm_c_epoch*.pt                  # checkpoints intermedios, no versionados
rm -f results/models/farm_b_best.pt                    # NO esta en git: git checkout no lo borra
```

**2 · Pasa siempre `--out` al preprocesado.** `scripts/preproc/run_preprocessing.py` escribe seis
ficheros por parque sin comprobar si existen, sin aviso y sin `--force`. **De los dos comandos posibles, el más corto es el destructivo.**
---

## Límites declarados

**Las cifras de referencia del *benchmark* no se recalculan desde este árbol.** El 0.6627 del parque
B y el 0.6364 del C salen de un repositorio auxiliar aparte, cuyo componente CAREBaseline reproduce
el autoencoder simple del artículo y verifica la implementación de la métrica.

Lo que sí viaja es la métrica, y es lo que hace válida la comparación. `src/care_official/` es una
implementación propia de la métrica CARE definida por Guck et al. (2024): se escribió y se validó en
aquel repositorio auxiliar con 82 pruebas unitarias, aquí solo se adaptaron los imports.

**El *ensemble* campeón se publicó con la calibración anterior del detector de cambio.** Durante el
desarrollo se identificó un sesgo en el remuestreo del calibrador y la corrección quedó registrada
por escrito antes de volver a medir. Bajo la calibración corregida el campeón pasa de 0.7694 a
0.7685, y se conservó la cifra original por consistencia con el resto de tablas. Ninguna conclusión
se ve afectada: el *ensemble* sigue ganando en las cinco semillas y el peor caso del espectro de
pesos mejora.

