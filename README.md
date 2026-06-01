# Highway Sign Detection — Detección de Paneles de Autopista mediante Visión Clásica

> Detección automática de paneles de señalización de autopista (regiones azules con borde blanco) en imágenes reales tomadas desde vehículos en circulación, usando exclusivamente técnicas de visión por computador clásica (sin aprendizaje automático).

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-informational.svg" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/status-academic%20project-lightgrey.svg" alt="Status">
</p>

-----

## Descripción general

Sistema de detección de paneles de señalización de autopista a partir de imágenes capturadas desde vehículos en movimiento. El reto consiste en localizar regiones rectangulares de fondo azul saturado y borde blanco en escenas reales con condiciones adversas (iluminación variable, oclusiones, paneles pequeños o lejanos, y otras superficies azules que generan falsos positivos).

El proyecto se desarrolló bajo una restricción deliberada: **resolver el problema sin aprendizaje automático**, empleando únicamente procesamiento de imagen clásico. Esto obliga a diseñar manualmente cada etapa del pipeline (extracción de regiones candidatas, filtrado geométrico, segmentación por color y puntuación), lo que constituye un ejercicio sólido de ingeniería de visión por computador y de comprensión de los algoritmos a bajo nivel.

Como aportación propia más allá del enunciado, se diseñó un **segundo detector independiente** basado en color y transformada de Hough, y un **detector combinado** que fusiona ambos enfoques mediante *Non-Maximum Suppression*.

> **Contexto:** proyecto académico desarrollado **en equipo de 3 personas** como Práctica 1 de la asignatura Visión Artificial (URJC, curso 2025/26).

-----

## Características principales

- **Tres detectores seleccionables** desde línea de comandos: `mser`, `hough` y `combinado`.
- **Detector MSER** con realce de contraste (CLAHE) y filtrado geométrico de regiones candidatas.
- **Detector propio basado en color + Hough** (iniciativa fuera del enunciado) que premia la presencia de estructura rectilínea.
- **Segmentación por color en espacio HSV** para aislar el azul saturado característico de los paneles.
- **Sistema de puntuación por F1-score** contra una máscara ideal normalizada.
- **Fusión de detecciones** mediante NMS con criterios de IoU e IoS para eliminar duplicados.
- **Evaluación cuantitativa** con métricas de precisión, recall y *Average Precision* (AP) a distintos umbrales de IoU.
- **CLI modular** con rutas y detector configurables.

-----

## Arquitectura del sistema

El sistema sigue un pipeline por etapas. El detector combinado ejecuta las dos ramas y fusiona sus salidas:

```
Imagen de entrada (BGR)
        │
        ├──────────────── Rama MSER ────────────────┐
        │                                            │
        │   BGR → escala de grises                   │
        │   CLAHE (clipLimit=2.0, tile 8×8)          │
        │   MSER_create + detectRegions              │
        │   Filtrado: tamaño · aspect ratio · ocupación
        │   Expansión de bbox (incluir borde blanco) │
        │   Máscara HSV de azul + F1-score (umbral 0.3)
        │                                            │
        ├──────────── Rama Color + Hough ────────────┤
        │                                            │
        │   Máscara HSV de azul                      │
        │   Morfología (cierre / apertura)           │
        │   findContours → boundingRect              │
        │   Canny (50,150) + HoughLinesP             │
        │   Conteo de líneas H/V (+0.05 si rectilíneo)
        │                                            │
        └────────────────┬───────────────────────────┘
                         │
              Fusión por NMS (IoU 0.3, IoS > 0.4)
                         │
              Detecciones finales (bbox + score)
```

Para una descripción detallada de cada etapa y de las decisiones de diseño, consulta [`ARCHITECTURE.md`](ARCHITECTURE.md).

-----

## Tecnologías utilizadas

|Categoría            |Tecnología                                                                  |
|---------------------|----------------------------------------------------------------------------|
|Lenguaje             |Python 3.10+                                                                |
|Visión por computador|OpenCV                                                                      |
|Cálculo numérico     |NumPy                                                                       |
|Técnicas             |MSER, CLAHE, segmentación HSV, morfología, Canny, transformada de Hough, NMS|
|Métricas             |Precisión, Recall, Average Precision (AP) a IoU 0.5 y 0.7                   |


> **Sin aprendizaje automático**, por requisito del enunciado. Todo el pipeline es procesamiento de imagen clásico.

-----

## Estructura del proyecto

```
.
├── main.py                              # Punto de entrada, pipeline y CLI
├── evaluar_resultados.py                # Evaluación: precisión, recall y AP
├── train_detection/                     # Imágenes de entrenamiento (97)
├── test_detection/                      # Imágenes de test (102)
├── resultado_imgs/                      # Imágenes con las detecciones dibujadas
├── resultado.txt                        # Detecciones generadas (nombre;x1;y1;x2;y2;tipo;score)
├── resultado_jmbuena_road_panels.txt    # Anotaciones / referencia de evaluación
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── .gitignore
└── LICENSE
```

-----

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/joel-domene/Proyecto1.git
cd Proyecto1

# 2. (Recomendado) Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

Contenido sugerido de `requirements.txt`:

```
opencv-python
numpy
```

-----

## Configuración

El sistema se configura mediante argumentos de línea de comandos:

|Argumento     |Descripción                         |Valores                       |
|--------------|------------------------------------|------------------------------|
|`--train_path`|Ruta a las imágenes de entrenamiento|Ruta a directorio             |
|`--test_path` |Ruta a las imágenes de test         |Ruta a directorio             |
|`--detector`  |Detector a utilizar                 |`mser` · `hough` · `combinado`|

Las anotaciones de *ground truth* / referencia se encuentran en `resultado_jmbuena_road_panels.txt`, y las detecciones generadas se vuelcan en `resultado.txt`, ambos con el formato:

```
nombre_imagen;x1;y1;x2;y2;tipo;score
```

-----

## Uso

```bash
# Detector MSER
python main.py --train_path train_detection --test_path test_detection --detector mser

# Detector propio (color + Hough)
python main.py --train_path train_detection --test_path test_detection --detector hough

# Detector combinado (recomendado: mejores resultados)
python main.py --train_path train_detection --test_path test_detection --detector combinado
```

-----

## Ejemplos de ejecución

```bash
$ python main.py --test_path test_detection --detector combinado

[INFO] Procesando 102 imágenes de test...
[INFO] Detector: combinado (MSER + Color/Hough + NMS)
[INFO] Evaluación a IoU > 0.5  -> Precisión: ~85%  Recall: ~65%  AP: 47.5
[INFO] Evaluación a IoU > 0.7  -> Precisión: ~80%  Recall: ~55%  AP: 28.2
[INFO] Resultados guardados en resultado_imgs/ y resultado.txt
```

> Salida ilustrativa. Los valores exactos dependen de la ejecución y de los parámetros.

-----

## Resultados visuales

El detector combinado localiza los paneles y muestra su puntuación de confianza:

![Detección de paneles de autopista con sus puntuaciones de confianza](docs/deteccion-ejemplo.png)

*Detección de tres paneles con sus scores (0.84, 0.80, 0.75) mediante el detector combinado.*

-----

## Decisiones técnicas relevantes

- **MSER como extractor de regiones candidatas:** los paneles son regiones de intensidad estable frente al fondo; MSER las captura bien sin necesidad de entrenamiento.
- **CLAHE en lugar de ecualización global:** el realce de contraste local mejora la separación del panel en escenas con iluminación heterogénea sin saturar el resto de la imagen.
- **Segmentación en HSV y no en RGB:** el canal de tono (H) aísla el azul de forma robusta frente a cambios de iluminación, algo difícil en RGB.
- **Expansión de la *bounding box*:** MSER tiende a capturar el interior azul; se expande la caja para incluir el borde blanco característico del panel.
- **Puntuación por F1-score contra máscara ideal:** normalizar la ventana a 40×80 y compararla con un patrón ideal permite una métrica de confianza interpretable (umbral 0.3).
- **Segundo detector por Hough:** la transformada de Hough aprovecha que los paneles tienen estructura rectilínea fuerte (bordes horizontales y verticales), aportando una señal complementaria a MSER.
- **Fusión por NMS con IoU e IoS:** combinar dos detectores genera duplicados; el NMS con criterios de solapamiento (IoU 0.3) y de inclusión (IoS > 0.4) los elimina y mejora la precisión global.

-----

## Retos encontrados y soluciones implementadas

|Reto                                         |Solución implementada                                                          |
|---------------------------------------------|-------------------------------------------------------------------------------|
|Iluminación variable entre imágenes          |Realce de contraste local con CLAHE antes de MSER                              |
|Falsos positivos en otras superficies azules |Filtrado geométrico (aspect ratio, tamaño, ocupación) + puntuación por F1-score|
|Paneles con interior detectado pero sin borde|Expansión de la *bounding box* para incluir el marco blanco                    |
|Duplicados al combinar dos detectores        |Fusión por NMS con umbrales de IoU e IoS                                       |
|Distinguir un panel real de una mancha azul  |Bonus por estructura rectilínea (líneas H/V vía Hough)                         |

-----

## Aprendizajes obtenidos

- Diseño de un **pipeline de visión por computador de extremo a extremo** sin recurrir a ML.
- Comprensión profunda de **MSER, CLAHE, segmentación HSV, morfología, Canny y la transformada de Hough**, y de cuándo aplicar cada técnica.
- Implementación de **métricas de evaluación de detección de objetos** (precisión, recall y AP a distintos umbrales de IoU).
- Manejo de la **fusión de detectores y NMS** como problema de ingeniería real.
- **Criterio para evaluar el coste/beneficio de la visión clásica frente al deep learning**, entendiendo por qué hoy el aprendizaje profundo es el estándar en este tipo de tareas.

-----

## Limitaciones conocidas

- Dificultad con **paneles pequeños o lejanos** (poca información de píxeles).
- Sensibilidad a la **oclusión parcial** de los paneles.
- **Falsos positivos** residuales en regiones azules grandes (cielo, vehículos, vallas).

![Ejemplo de detección con falsos positivos y fragmentación del panel](docs/deteccion-limitaciones.png)

*Caso difícil: aparecen falsos positivos en el fondo y el panel grande se fragmenta en varias cajas.*

-----

## Mejoras futuras

- **Aprendizaje profundo:** detectores como YOLO o Faster R-CNN superarían el rendimiento del pipeline clásico (las referencias del enunciado ya apuntan a un margen de mejora claro).
- **Búsqueda multiescala** para mejorar la detección de paneles lejanos.
- **Tracking temporal** aprovechando que las imágenes provienen de secuencias de vídeo.
- **Segmentación semántica** para una localización a nivel de píxel.

-----

## Resultados

Métricas del **detector combinado** sobre el conjunto de test (102 imágenes):

|Umbral   |Precisión|Recall|AP  |
|---------|---------|------|----|
|IoU > 0.5|~85%     |~65%  |47.5|
|IoU > 0.7|~80%     |~55%  |28.2|


> Referencia de los profesores: AP 68.7 (IoU 0.5) / 60.9 (IoU 0.7).

Curvas precisión-recall, comparando el detector propio con la referencia:

![Curva precisión-recall a IoU > 0.5](docs/pr-curve-iou50.png)

![Curva precisión-recall a IoU > 0.7](docs/pr-curve-iou70.png)

-----

## Contexto académico

- **Asignatura:** Visión Artificial — Universidad Rey Juan Carlos (URJC).
- **Curso:** 2025/26 — Práctica 1 grupal.
- **Restricción:** resolución sin aprendizaje automático.

-----

## Licencia

Distribuido bajo licencia MIT. Consulta [`LICENSE`](LICENSE) para más información.
