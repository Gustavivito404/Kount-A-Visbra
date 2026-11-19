# Kount-A — Versión 14 (YOLO + BBox manual/múltiple)

## Requisitos

- Python 3.10, 3.11 o 3.12 recomendado (evita versiones recién publicadas si PyTorch/Ultralytics no tienen wheels compatibles).
- Windows 11 o Linux; interfaz Tkinter instalada.
- PyTorch y torchvision compatibles con la GPU si deseas usar CUDA.
- Internet la **primera vez** que descargues `yolo11n.pt` (o el peso elegido).

## Instalación y ejecución

Desde la carpeta que contiene `src`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.gui_app
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.gui_app
```

Si `torch.cuda.is_available()` imprime `False`, instala la versión de PyTorch adecuada para tu versión de CUDA desde https://pytorch.org/get-started/locally/.

## Estructura

```text
KountA_YOLO_v14/
  src/
    __init__.py
    project_core.py       # Clases, lectura/escritura de anotaciones, BBox gris
    augment_core.py       # Transformaciones PyTorch que actualizan todas las BBox
    yolo_core.py          # Split, exportación YOLO, entrenamiento, predicción
    camera_utils.py       # Búsqueda de webcams y captura de un frame
    gui_app.py            # GUI completa de cuatro pestañas
  requirements.txt
  README.md
  input/                    # Opcional: también puedes elegir otra carpeta
  output/                   # Se genera automáticamente
```

Las imágenes pueden estar organizadas por carpetas de clase, por ejemplo:

```text
input/
  tornillos/
  tuercas/
  rondanas/
  mezclas/                 # fotografías con distintas clases en la misma toma
```

## Flujo recomendado

1. **BBox y etiquetado**. Elige input/output arriba. Selecciona una fotografía. Para las primeras etiquetas puedes pulsar «Proponer BBox» por gris, o dibujar con mouse haciendo clic y arrastrando en el lienzo. Ajusta la clase por cada rectángulo. «Nueva» agrega otra caja; «Reemplazar seleccionada» reubica la caja seleccionada. Usa la lista lateral, «Aplicar clase», «Eliminar BBox», y «Guardar BBox».
2. **Anotación multiobjeto**. En una foto con muchas piezas, dibuja una caja por cada pieza y asigna su clase: tornillo, tuerca, rondana o versiones oxidadas. La foto conserva una lista `objects` en `input/.kounta_annotations/`.
3. **Aumentación y revisión**. Configura brillo, contraste, ruido, número de copias, etc. Elige «Generar». Inspecciona imágenes originales y aumentadas, cada una con todas las BBox dibujadas.
4. **Entrenamiento**. Usa `yolo11n.pt` inicialmente o elige un archivo `best.pt` obtenido de YOLO anteriormente. Ajusta épocas, batch, resolución, learning rate, patience, seed, fracciones de validación/test. Pulsar «Entrenar» exporta y lanza la red.
5. **Pruebas**. Carga `best.pt`; abre fotografía o busca cámaras mediante botón, elige una del menú, captura y cuenta.
6. **Fine-tuning**. Usa tu `best.pt` anterior como modelo inicial; incorpora fotos nuevas con anotaciones revisadas **además** de conservar las viejas.

## Compatibilidad con versiones previas

- **No se borra tu dataset original**. Las anotaciones manuales se escriben en `.kounta_annotations` dentro del input.
- Si existen etiquetas antiguas en `output/original_labels/<clase>/<archivo>.json`, se recupera automáticamente la primera BBox cuando no existe anotación manual (para clases reconocidas).
- El antiguo detector Faster R-CNN (`.pth`) **no puede** convertirse en modelo YOLO simplemente cambiando la extensión; no se reutilizan sus pesos en el entrenamiento YOLO. Sí se pueden reutilizar imágenes y BBox previas.
- El modo `rgb_range` fue eliminado de la versión nueva; queda propuesta inicial mediante escala de grises y edición manual.

## Dataset YOLO

`output/yolo_dataset/dataset.yaml` y carpetas `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, `labels/test`.

**Separación sin fuga**: cada foto original recibe su grupo train/val/test y se registra en `output/split_assignments.json`. Las aumentaciones de esa foto solo se incluyen si pertenece a train. Validación y prueba conservan originales sin augmentation. Si cambias seed/porcentajes, mueve o borra ese JSON deliberadamente antes de reexportar, pues eso cambia el protocolo de evaluación.

**Importante para tesis**: no uses las mismas fotos del test final para tomar decisiones de hiperparámetros. Reserva también lotes reales totalmente nuevos para evaluación final de conteo.

## Pesos, informes y curvas

- `output/augmentation_report.json`: ajustes y cantidades creadas.
- `output/yolo_dataset/export_report.json`: tamaños por conjunto y clases.
- `output/models/runs/kounta_.../`: entrenamiento Ultralytics, métricas, gráficas y `weights/best.pt`.
- `output/models/kounta_..._best.pt`: copia del modelo ganador que puedes usar en Prueba.
- `output/models/kounta_..._report.json`: datos de entrenamiento útiles para documentar la tesis.

## Recomendaciones y límites

- YOLO aplica NMS de manera interna. Ajusta `IoU NMS` y confianza en la pestaña Prueba; no son sustituto de anotaciones correctas ni ejemplos multiobjeto.
- El modo gris es solo un **borrador**: fondos, sombras y contactos entre piezas pueden generar cajas incorrectas. Revísalas.
- Una foto con distintas clases requiere una caja y clase por pieza.
- Las clases oxidadas están definidas, pero no se entrenarán de manera fiable hasta incorporar fotos reales de esas clases.
- En una GPU portátil con VRAM limitada, comienza con `yolo11n.pt`, `imgsz=640`, `batch=2` o `4` y reduce el batch si aparece error de memoria CUDA.
- La detección de cámaras muestra dispositivos operativos, pero según el controlador de Windows las webcams pueden no publicar sus nombres comerciales; por eso se muestra una lista identificable con resolución e índice interno, **sin obligarte a escribir un número**.
- ArUco permanece como módulo independiente: el modelo decide clase/condición, la calibración posterior estima tamaño.
- La interfaz muestra imágenes aumentadas y sus BBox; no representa una garantía de desempeño hasta que valides sobre imágenes inéditas.

## Verificación local y límites de las pruebas

Desde la carpeta principal ejecuta:

```powershell
python -m src.smoke_test
```

Esta prueba genera fotografías de ejemplo temporales, comprueba etiquetas de **varios objetos**, augmentación y exportación YOLO sin fuga entre conjuntos. La interfaz también fue comprobada con Tkinter, incluida la creación y almacenamiento de cajas con el mouse. El entrenamiento **real** de Ultralytics no se ejecutó en el entorno donde se armó este paquete, porque allí no estaba instalada la librería `ultralytics` ni se contaba con un entrenamiento real de tus fotografías; debes instalar dependencias y realizar la primera ejecución en tu equipo.

Durante entrenamiento, la GUI muestra el **batch actual** y, al completar cada época, las métricas de validación que proporcione YOLO.

## Licencias para un producto comercial

La implementación usa la librería **Ultralytics YOLO**. Según las condiciones publicadas por Ultralytics, el uso de su código/modelos con licencia **AGPL-3.0** puede obligar a ofrecer el código fuente correspondiente del producto derivado; Ultralytics también ofrece licencia **Enterprise** para modalidades comerciales con otras condiciones. Antes de integrar la red a una máquina comercial de distribución cerrada, **revisa las condiciones de licencia** con tu equipo. Información oficial: https://docs.ultralytics.com/es/help/contributing/ y https://docs.ultralytics.com/es/ .
