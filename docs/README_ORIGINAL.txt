Versión 8 del proyecto:
- Corrige el guardado automático por clase
- Corrige la persistencia del JSON
- Ya no se arrastra el umbral de una clase a otra al cambiar en el combobox
- Crea/actualiza input/augment_thresholds_config.json al cambiar selección, perder foco, actualizar preview y generar
- Si faltan clases al cargar el JSON, pregunta si se eliminan o se conservan

Ejecución:
    python src/gui_app.py

Dependencias:
    pip install torch torchvision pillow tqdm numpy
