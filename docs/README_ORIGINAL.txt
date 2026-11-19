Proyecto base: GUI de augmentación con Tkinter + bounding boxes automáticas desde fondo blanco

Archivos:
- src/gui_app.py
- src/augment_core.py
- src/bbox_utils.py

Instalación sugerida:
    pip install torch torchvision pillow tqdm numpy

Ejecución:
    python src/gui_app.py

Notas:
1) El detector de bounding boxes asume fondo blanco o casi blanco.
2) La extracción del bbox original se hace por umbral: todo pixel no blanco se considera objeto.
3) Las transformaciones se aplican de forma conjunta a imagen + bounding box usando torchvision.transforms.v2 y tv_tensors.BoundingBoxes.
4) Se generan JSON para originales y augmentadas con bbox en formatos XYXY, XYWH y YOLO normalizado.
5) Si luego quieres entrenar un detector, estos JSON se pueden convertir a TXT estilo YOLO o a COCO.
