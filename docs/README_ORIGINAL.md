# Augment GUI Project v12

La interfaz se centra ahora exclusivamente en el flujo de detección y conteo.

## Pestañas visibles

### 1. Dataset / Augmentación
Conserva el flujo de augmentación y agrega dos métodos para obtener las bounding boxes:

- `white`: umbral de fondo blanco.
- `rgb_range`: rango independiente por canal R, G y B.

En RGB se configuran:

```text
R min / R max
G min / G max
B min / B max
```

La vista previa de la imagen original permite ajustar el rango antes de generar el dataset.

### 2. Entrenamiento detector
Incluye únicamente lo relacionado con entrenamiento:

- dataset de salida
- modelo `.pth`
- épocas
- batch size
- learning rate
- porcentaje de validación
- seed
- pesos COCO preentrenados
- barra de progreso
- mejor modelo
- log grande con avance por épocas, precision, recall y F1

### 3. Prueba de conteo
La prueba se separó del entrenamiento para poder visualizar mucho mejor la imagen.

```text
┌─────────────────────────────────────┬──────────────────────┐
│                                     │ Resultado             │
│                                     │                      │
│          IMAGEN GRANDE              │ Total                │
│      + bounding boxes               │ Conteo por clase     │
│                                     │ Confianzas           │
│                                     │ BBoxes               │
└─────────────────────────────────────┴──────────────────────┘
```

Permite:

- seleccionar una fotografía guardada
- capturar un frame de webcam
- capturar y contar
- ajustar el umbral de confianza
- ver bounding boxes
- ver conteo total y por clase
- ver confianza y coordenadas de cada detección

La webcam continúa siendo opcional.

## Ejecución

```bash
pip install -r requirements.txt
python -m src.gui_app
```
