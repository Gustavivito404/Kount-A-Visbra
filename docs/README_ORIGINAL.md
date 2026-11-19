# Augment GUI Project v11 - Pruebas con monedas

Esta revisión mantiene la v11 y mejora el flujo para hacer pruebas de conteo
seleccionando imágenes desde la computadora, sin necesidad de tener webcam.

## Estructura

```text
augment_gui_project_v11_monedas/
├── src/
│   ├── __init__.py
│   ├── gui_app.py
│   ├── augment_core.py
│   ├── bbox_utils.py
│   ├── classifier_core.py
│   └── detection_core.py
├── README.md
└── requirements.txt
```

## Prueba mínima recomendada con monedas

### 1. Crear clase

```text
input/
└── moneda/
    ├── moneda_01.jpg
    ├── moneda_02.jpg
    ├── moneda_03.jpg
    └── ...
```

Las imágenes de entrenamiento deberían contener inicialmente UNA moneda
por fotografía sobre un fondo lo más uniforme posible.

### 2. Generar augmentation

Usa la pestaña:

`1. Dataset / Augmentación`

El programa genera las imágenes y bounding boxes.

### 3. Entrenar detector

Usa la pestaña:

`3. Detección / Conteo`

Valores iniciales sugeridos:

```text
Épocas: 10
Batch: 2
Learning rate: 0.0005
Validación: 0.20
Seed: 42
Pesos COCO preentrenados: activado
```

### 4. Hacer fotografía nueva

Toma una fotografía que NO haya sido usada en entrenamiento.

Ejemplo:

```text
foto_prueba.jpg
- 5 monedas
- relativamente separadas
- buena iluminación
- fondo uniforme
```

### 5. Probar

En `3. Detección / Conteo`:

1. Seleccionar imagen
2. Elegir `foto_prueba.jpg`
3. Presionar `Contar objetos`

La GUI mostrará:
- bounding boxes
- confianza de cada detección
- cantidad por clase
- total detectado

## Webcam

La sección de webcam se conserva porque se utilizará después, pero es
completamente opcional. Para las pruebas actuales basta con seleccionar
fotografías guardadas.

## Importante

El detector puede intentar encontrar varias monedas aunque el dataset inicial
tenga una moneda por fotografía. Sin embargo, esto es una prueba de concepto.

Para mejorar el conteo posteriormente convendrá entrenar también con imágenes
que contengan varias monedas y una bounding box por cada moneda.

## Ejecutar

Desde la raíz:

```bash
pip install -r requirements.txt
python -m src.gui_app
```
