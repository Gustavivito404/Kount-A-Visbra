Versión 9 del proyecto:
- Se mantiene el JSON de umbrales dentro de input porque pertenece al dataset
- Se agrega un JSON de sesión llamado ui_session_config.json junto al proyecto de la app
- La app recuerda la última carpeta input y la última carpeta output
- Al abrir la interfaz, si esas rutas siguen existiendo, se cargan automáticamente
- También se recargan las configuraciones de umbrales del input recordado

Ejecución:
    python src/gui_app.py

Dependencias:
    pip install torch torchvision pillow tqdm numpy
