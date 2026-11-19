# Versiones de Visbra · Trazabilidad documental

## Convención solicitada

El formato `vETAPA.VERSION.REVISION` utiliza **tres números**:

- `ETAPA`: 1 = desarrollo inicial (v4 a v10), 2 = detección y entrenamiento (v11 a v14), 3 = integración final (v15 y v16).
- `VERSION`: número original (4, 5, ..., 16) sin renumeración.
- `REVISION`: 0 para la entrega base disponible; 1, 2, ... para ajustes documentados por archivos de actualización o comparaciones de versiones.

**No es SemVer estándar**; el primer número significa etapa documental, no ruptura de API.

## Limitaciones y decisiones verificadas

1. **v4 asignada provisionalmente**: `augment_gui_project/` carece de numeración explícita. Su README y sus funciones anteriores a v5 permiten asociarla de manera razonable con v4, pero debe confirmarlo el autor.
2. **v7 no aparece** como snapshot en `Avance cronologico.zip`. El salto `v1.6.0` → `v1.8.0` es deliberado; no se ha fabricado v7.
3. `augment_dataset/` es un antecedente previo sin número y se archiva fuera de la numeración.
4. Se recuperaron **modificaciones puntuales de v15 y v16** de ZIP de ajustes disponibles en la conversación; su orden documental no necesariamente refleja las fechas originales de cada cambio.
5. **Credenciales, hashes de usuarios, rutas personales, sesiones, datasets, pesos y salidas** fueron excluidos de todas las versiones públicas. El ZIP original permanece intacto.
6. Todos los commits del bundle son **reconstrucciones de documentación**, no commits originales ni evidencia de fecha o autoría histórica.

## Inventario y comparaciones de código

| Etiqueta | Desarrollo principal | Archivo de origen | Variación A/M/R* |
|---|---|---|---|
| `v1.4.0` | Primer entorno gráfico de aumento de imágenes y BBox sobre fondo claro. | augment_gui_project | 5/0/0 |
| `v1.5.0` | Ventana de notas, recarga de vista previa y configuración de umbrales por clase. | augment_gui_project_v5 | 1/4/1 |
| `v1.6.0` | Configuración de umbrales por clase integrada en el panel izquierdo. | augment_gui_project_v6 | 0/2/0 |
| `v1.8.0` | Persistencia automática de umbrales y corrección del cambio de clase. | augment_gui_project_v8 | 0/3/0 |
| `v1.9.0` | Guardado de última sesión y recuperación de carpetas de entrada/salida. | augment_gui_project_v9 | 0/2/0 |
| `v1.10.0` | Se incorpora entrenamiento y prueba de una CNN para clasificación de imagen completa. | augment_gui_project_v10 | 3/3/1 |
| `v2.11.0` | Se incorpora detección de múltiples objetos y pruebas con monedas. | augment_gui_project_v11 | 1/4/0 |
| `v2.12.0` | Se separan entrenamiento del detector y pruebas de conteo; se añaden umbrales por RGB. | augment_gui_project_v12 | 0/4/1 |
| `v2.13.0` | Se amplían ajustes de aumento de datos: iluminación, color y ruido visual. | augment_gui_project_v13 | 1/3/0 |
| `v2.14.0` | Transición al flujo YOLO, etiquetado multiobjeto y administración de BBox. | KountA_YOLO_v14 | 4/5/3 |
| `v3.15.0` | Se incorporan perfiles, roles, HMI ampliada y medición mediante ArUco. | KountA_YOLO_v15.zip | 4/6/0 |
| `v3.15.1` | Ajustes de integración de perfiles, roles y control de vistas. | KountA_v15_ajustada_completa.zip | 0/5/0 |
| `v3.15.2` | Ajustes de menús desplegables y organización de la interfaz. | KountA_v15_completa_ajuste_menus.zip | 0/1/0 |
| `v3.15.3` | Refinamientos de v15, incluida visibilidad de BBox y actualización del panel de prueba. | Avance cronologico.zip::KountA_YOLO_v15 | 0/2/0 |
| `v3.16.0` | Primer sistema v16: cámara en vivo, secuencias programables y comunicación ESP32. | KountA_YOLO_v16.zip | 6/4/0 |
| `v3.16.1` | Ajustes de conexión y comprobación TCP con ESP32. | KountA_v16_ajuste_conexion_ESP32_solo_archivos.zip | 0/3/0 |
| `v3.16.2` | Pausa de 3 segundos entre instrucciones de trabajo. | KountA_v16_pausa_3s_solo_sequence_ui.zip | 0/1/0 |
| `v3.16.3` | Saturación visual por píxeles ocupados en la zona ArUco. | KountA_v16_saturacion_aruco_solo_archivo.zip | 0/1/0 |
| `v3.16.4` | Ventana de resultados ampliados, pausa configurable y logotipo con mayor contraste. | Avance cronologico.zip::KountA_YOLO_v16 | 2/4/0 |

*A/M/R: número de archivos añadidos, modificados y retirados frente a la etiqueta anterior; no representa número de líneas ni calidad funcional.*

## Cambios desglosados

### v1.4.0 — Primer entorno gráfico de aumento de imágenes y BBox sobre fondo claro.
Origen: `augment_gui_project`. Entrega base sin numeración explícita: asignación v4 inferida por orden y funciones frente a v5.

- Añadidos: `README.txt`, `docs/ANTECEDENTE_PREVIO_A_LA_GUI.md`, `src/augment_core.py`, `src/bbox_utils.py`, `src/gui_app.py`
- Modificados: ninguno
- Retirados: ninguno

### v1.5.0 — Ventana de notas, recarga de vista previa y configuración de umbrales por clase.
Origen: `augment_gui_project_v5`. READMEs y código del directorio v5.

- Añadidos: `requirements.txt`
- Modificados: `README.txt`, `src/augment_core.py`, `src/bbox_utils.py`, `src/gui_app.py`
- Retirados: `docs/ANTECEDENTE_PREVIO_A_LA_GUI.md`

### v1.6.0 — Configuración de umbrales por clase integrada en el panel izquierdo.
Origen: `augment_gui_project_v6`. README y código del directorio v6.

- Añadidos: 
- Modificados: `README.txt`, `src/gui_app.py`
- Retirados: ninguno

### v1.8.0 — Persistencia automática de umbrales y corrección del cambio de clase.
Origen: `augment_gui_project_v8`. No se recibió v7 como carpeta independiente.

- Añadidos: 
- Modificados: `README.txt`, `src/augment_core.py`, `src/gui_app.py`
- Retirados: ninguno

### v1.9.0 — Guardado de última sesión y recuperación de carpetas de entrada/salida.
Origen: `augment_gui_project_v9`. README del directorio v9.

- Añadidos: 
- Modificados: `README.txt`, `src/gui_app.py`
- Retirados: ninguno

### v1.10.0 — Se incorpora entrenamiento y prueba de una CNN para clasificación de imagen completa.
Origen: `augment_gui_project_v10`. El peso entrenado .pth se excluye del paquete público.

- Añadidos: `README.md`, `src/__init__.py`, `src/classifier_core.py`
- Modificados: `requirements.txt`, `src/augment_core.py`, `src/gui_app.py`
- Retirados: `README.txt`

### v2.11.0 — Se incorpora detección de múltiples objetos y pruebas con monedas.
Origen: `augment_gui_project_v11`. Detección basada en código detection_core.py y README v11.

- Añadidos: `src/detection_core.py`
- Modificados: `README.md`, `requirements.txt`, `src/__init__.py`, `src/gui_app.py`
- Retirados: ninguno

### v2.12.0 — Se separan entrenamiento del detector y pruebas de conteo; se añaden umbrales por RGB.
Origen: `augment_gui_project_v12`. README v12 y código.

- Añadidos: 
- Modificados: `README.md`, `src/augment_core.py`, `src/bbox_utils.py`, `src/gui_app.py`
- Retirados: `src/classifier_core.py`

### v2.13.0 — Se amplían ajustes de aumento de datos: iluminación, color y ruido visual.
Origen: `augment_gui_project_v13`. README_CAMBIOS.md; el README principal conserva título v12.

- Añadidos: `src/README_CAMBIOS.md`
- Modificados: `src/augment_core.py`, `src/detection_core.py`, `src/gui_app.py`
- Retirados: ninguno

### v2.14.0 — Transición al flujo YOLO, etiquetado multiobjeto y administración de BBox.
Origen: `KountA_YOLO_v14`. Los pesos preentrenados y entrenados se excluyen.

- Añadidos: `src/camera_utils.py`, `src/project_core.py`, `src/smoke_test.py`, `src/yolo_core.py`
- Modificados: `README.md`, `requirements.txt`, `src/__init__.py`, `src/augment_core.py`, `src/gui_app.py`
- Retirados: `src/README_CAMBIOS.md`, `src/bbox_utils.py`, `src/detection_core.py`

### v3.15.0 — Se incorporan perfiles, roles, HMI ampliada y medición mediante ArUco.
Origen: `KountA_YOLO_v15.zip`. Primera entrega v15 verificable, recuperada del ZIP previo.

- Añadidos: `assets/kounta_logo.png`, `src/aruco_core.py`, `src/auth_store.py`, `src/profile_store.py`
- Modificados: `README.md`, `requirements.txt`, `src/__init__.py`, `src/camera_utils.py`, `src/gui_app.py`, `src/project_core.py`
- Retirados: ninguno

### v3.15.1 — Ajustes de integración de perfiles, roles y control de vistas.
Origen: `KountA_v15_ajustada_completa.zip`. Se conserva íntegro el código de la entrega intermedia.

- Añadidos: 
- Modificados: `README.md`, `src/auth_store.py`, `src/gui_app.py`, `src/profile_store.py`, `src/yolo_core.py`
- Retirados: ninguno

### v3.15.2 — Ajustes de menús desplegables y organización de la interfaz.
Origen: `KountA_v15_completa_ajuste_menus.zip`. ZIP de actualización intermedia con nota de cambio.

- Añadidos: 
- Modificados: `src/gui_app.py`
- Retirados: ninguno

### v3.15.3 — Refinamientos de v15, incluida visibilidad de BBox y actualización del panel de prueba.
Origen: `Avance cronologico.zip::KountA_YOLO_v15`. Código final v15 del archivo cronológico; mismos bytes que v15(1).zip.

- Añadidos: 
- Modificados: `README.md`, `src/gui_app.py`
- Retirados: ninguno

### v3.16.0 — Primer sistema v16: cámara en vivo, secuencias programables y comunicación ESP32.
Origen: `KountA_YOLO_v16.zip`. ZIP de entrega inicial de v16.

- Añadidos: `src/esp32_link.py`, `src/live_camera.py`, `src/live_ui.py`, `src/sequence_core.py`, `src/sequence_ui.py`, `src/smoke_v16.py`
- Modificados: `README.md`, `requirements.txt`, `src/camera_utils.py`, `src/gui_app.py`
- Retirados: ninguno

### v3.16.1 — Ajustes de conexión y comprobación TCP con ESP32.
Origen: `KountA_v16_ajuste_conexion_ESP32_solo_archivos.zip`. Se superponen solo los archivos especificados en el ZIP de parche.

- Añadidos: 
- Modificados: `src/esp32_link.py`, `src/gui_app.py`, `src/sequence_ui.py`
- Retirados: ninguno

### v3.16.2 — Pausa de 3 segundos entre instrucciones de trabajo.
Origen: `KountA_v16_pausa_3s_solo_sequence_ui.zip`. Parche específico del ejecutor de secuencias.

- Añadidos: 
- Modificados: `src/sequence_ui.py`
- Retirados: ninguno

### v3.16.3 — Saturación visual por píxeles ocupados en la zona ArUco.
Origen: `KountA_v16_saturacion_aruco_solo_archivo.zip`. Parche específico de aruco_core.py.

- Añadidos: 
- Modificados: `src/aruco_core.py`
- Retirados: ninguno

### v3.16.4 — Ventana de resultados ampliados, pausa configurable y logotipo con mayor contraste.
Origen: `Avance cronologico.zip::KountA_YOLO_v16`. Estado final de v16 del archivo cronológico; verificado contra el último parche.

- Añadidos: `assets/1.png`, `src/prb_aruco.py`
- Modificados: `assets/kounta_logo.png`, `src/gui_app.py`, `src/sequence_core.py`, `src/sequence_ui.py`
- Retirados: ninguno

## Seguridad del material

Se excluyeron 497 entradas durante la revisión de los archivos base y parches. Principales motivos:

- imágenes/datos derivados de pruebas: 382 archivos
- caché de Python: 75 archivos
- configuración privada, credenciales o sesión: 28 archivos
- perfiles y resultados locales: 5 archivos
- pesos, entornos o salidas: 3 archivos
- binario de modelo o datos: 2 archivos
- referencia gráfica redundante: 2 archivos

Los pesos excluidos de una versión no son recuperables mediante este paquete público. Tampoco se incluyeron imágenes privadas para entrenar el detector. Las dependencias y los modelos que se permitan usar deben documentarse por separado antes de ofrecer una entrega ejecutable.
