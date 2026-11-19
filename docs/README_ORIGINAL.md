# Augment GUI Project v10

Estructura del proyecto:

```text
augment_gui_project_v10/
├── src/
│   ├── __init__.py
│   ├── gui_app.py
│   ├── augment_core.py
│   ├── bbox_utils.py
│   └── classifier_core.py
├── README.md
└── requirements.txt
```

Todos los archivos Python se encuentran dentro de `src/`.

## Instalación

Desde la carpeta raíz del proyecto:

```bash
pip install -r requirements.txt
```

## Ejecutar la interfaz

### Opción recomendada

Desde la carpeta `augment_gui_project_v10`:

```bash
python -m src.gui_app
```

### También funciona

```bash
python src/gui_app.py
```

No uses imports como:

```python
from augment_gui_project_v10.src.augment_core import ...
```

porque el nombre de la carpeta del proyecto no necesita estar instalado como paquete.

## Pestañas

### 1. Dataset / Augmentación
Mantiene el flujo de la v9:
- originales
- imágenes augmentadas
- bounding boxes
- configuración por clase

### 2. Red neuronal / Prueba
Usa:
- `output/original_images`
- `output/augmented_images`

para entrenar una CNN de clasificación.

Guarda por defecto:

```text
output/models/classifier_best.pth
```

Después permite seleccionar una imagen completa y devuelve:
- clase predicha
- confianza de la clase principal
- porcentaje para cada clase

Esta versión todavía es de **clasificación de imagen completa**.
No realiza conteo ni detección de varios objetos.
