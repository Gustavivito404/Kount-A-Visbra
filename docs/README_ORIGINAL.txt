Versión 5 del proyecto:
- Las notas ya no saturan la interfaz: ahora se abren en una ventana aparte
- Entre las dos visualizaciones centrales se agregó el botón "Actualizar previsualización original"
- Se agregó una ventana para configurar umbrales por clase
- La clase se detecta por la primera carpeta dentro de input
- Los parámetros por clase sobrescriben a los globales
- Los labels JSON guardan también el threshold y min_fg usados

Ejecución:
    python src/gui_app.py

Dependencias:
    pip install torch torchvision pillow tqdm numpy
