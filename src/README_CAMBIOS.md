# Actualización de Augmentación

Reemplaza estos tres archivos dentro de tu carpeta `src/`:

- `augment_core.py`
- `bbox_utils.py`
- `gui_app.py`

## Nuevos controles

Se mantienen los modos de BBox `white` y `rgb_range`.

Se agregan:

- Brillo (default `0.30`)
- Contraste (default `0.20`)
- Saturación (default `0.08`)
- Hue (default `0.01`)
- Sal y pimienta activable
- Probabilidad de sal y pimienta (default `0.30`)
- Cantidad de sal y pimienta (default `0.003`)

Interpretación:

- Brillo `0.30`: aproximadamente de 70% a 130% del brillo original.
- Probabilidad `0.30`: alrededor del 30% de las copias recibe ruido.
- Cantidad `0.003`: aproximadamente 0.3% de los píxeles se convierten en sal/pimienta.

El ruido se aplica después de las transformaciones geométricas para no
desajustar las bounding boxes.

Los valores se guardan en `augment_thresholds_config.json`.
