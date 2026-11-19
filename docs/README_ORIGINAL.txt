Versión 6 del proyecto:
- Las notas siguen en una ventana aparte
- El botón de actualizar previsualización original queda entre ambas visualizaciones
- La configuración de umbrales por clase ya no abre una ventana aparte
- Ahora se edita en el mismo panel izquierdo con un selector tipo desplegable
- Puedes elegir:
  - General / default
  - cada clase detectada en input
- Los cuadros de umbral y mínimo de pixeles cambian según la selección actual
- Puedes guardar los valores específicos de cada selección desde el mismo lugar

Ejecución:
    python src/gui_app.py

Dependencias:
    pip install torch torchvision pillow tqdm numpy
