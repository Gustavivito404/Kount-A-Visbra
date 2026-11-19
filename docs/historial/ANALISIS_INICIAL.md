# Kount-A / Visbra — historial técnico y tres avances para GitHub

**Base documental:** archivos ZIP recuperados y revisados. Los cambios de v4–v9 no se describen como hechos comprobados sin el paquete correspondiente.

## Inventario comprobado

| Versión | Respaldo disponible | Funcionalidad identificada |
|---|---|---|
| v4–v6 | No | Pendiente de recuperar. |
| v7 | No ZIP completo | Antecedentes de autoguardado y persistencia de umbrales: **pendiente de corroboración con código**. |
| v8 | No ZIP completo | Antecedentes de corrección de JSON y umbrales por clase: **pendiente de corroboración**. |
| v9 | No ZIP completo | Antecedentes de persistencia de rutas input/output: **pendiente de corroboración**. |
| v10 | `augment_gui_project_v10.zip` | Generación/aumento de imágenes y CNN de clasificación de imágenes completas con PyTorch; aún sin conteo multiobjeto. |
| v11 | `augment_gui_project_v11.zip` | Detector Faster R-CNN y módulo de detección/conteo con fotografía o captura puntual de webcam. |
| v12 | `augment_gui_project_v12.zip` | Interfaz centrada en detección/conteo; umbral blanco o intervalo RGB; entrenamiento y prueba organizados por pestañas. |
| v13 | `augment_v13_visual_noise.zip` | Cambios en augmentación: brillo, contraste, saturación, matiz y ruido sal-pimienta. Es un parche, no un proyecto completo. |
| v14 | `KountA_YOLO_v14.zip` | Migración a Ultralytics YOLO; anotación BBox múltiple y multiclase; exportación YOLO, entrenamiento e inferencia. |
| v15 | `KountA_YOLO_v15(1).zip` y parches | Perfiles independientes, usuarios/roles, interfaz por permisos, temas visuales y ArUco espacial/dimensional; ajustes de edición BBox y vista de conteo. |
| v16 | `KountA_YOLO_v16(1).zip` y parches de v16 | Video continuo, inspección manual/automática, secuencias de motor/LED/cámara, TCP ESP32, editor por perfil, guardado de distribución, saturación ArUco y ajustes de interfaz. |

> **Alcance de esta cronología:** identifica cambios del *software*; no constituye por sí sola evidencia de pruebas físicas, desempeño de IA ni validación experimental.

## Avance 1. Exploración y preparación del procesamiento visual (v4–v10; inicial pendiente)

Las primeras iteraciones de Visbra se orientaron a establecer herramientas de adquisición, etiquetado y aumento de imágenes para estudiar la clasificación visual de piezas. De las versiones 4 a 9 falta verificar el código original. En la versión 10 ya se observa un módulo de clasificación mediante CNN, capaz de estimar una categoría para la imagen completa, pero sin identificar simultáneamente varias piezas dentro de una escena.

**Evidencia disponible:** ZIP v10. **Evidencia pendiente:** ZIP v4 a v9. No atribuir características exactas a v4–v6 sin revisar su código.

## Avance 2. Detección multiobjeto y entrenamiento (v11–v14)

La versión 11 incorporó detección y conteo mediante Faster R-CNN; la versión 12 organizó el flujo en pestañas de datos, entrenamiento y pruebas, con herramientas de segmentación por umbral e intervalos RGB. Posteriormente se ampliaron las transformaciones de datos y la versión 14 integró YOLO, con anotación manual multiclase, exportación de conjuntos de datos, entrenamiento y conteo individual de objetos detectados.

**Evidencia disponible:** ZIP v11, v12, parche v13 y ZIP v14.

## Avance 3. Integración y operación del prototipo (v15–v16)

La versión 15 incorporó administración de perfiles, roles de usuario y funciones de ArUco para referencia espacial y medición. La versión 16 amplió la aplicación con adquisición continua desde webcam, identificación manual o automática, secuencias configurables de motor, iluminación y cámara, comunicación TCP con ESP32, guardado de fotografías de pruebas y visualización de métricas de ocupación. Esta etapa integra las funciones de ingeniería y operación en la misma interfaz, aunque requiere pruebas físicas para validar su desempeño definitivo.

**Evidencia disponible:** ZIP v15, ZIP v16 y archivos de actualización.

## Cómo publicarlo en GitHub sin inventar una historia de commits

1. Crear un repositorio **privado** de trabajo si aún contiene datos de prueba, usuarios, rutas reales o fotografías del proyecto.
2. Publicar únicamente una copia limpia del código de cada versión disponible. No subir directamente ZIP de instalación sin revisar.
3. Para cada respaldo verificable, usar una carpeta `releases-sources/vXX/` o, preferiblemente, una etiqueta Git `vXX` en el commit que contiene **exactamente** ese estado del código. Las etiquetas se crearán en orden, sin suponer fechas pasadas que no se conocen.
4. No fabricar commits retroactivos ni simular que se usó Git históricamente si el desarrollo ocurrió fuera del repositorio. Se puede indicar que se trata de una **reconstrucción documental basada en respaldos**.
5. Agregar una nota corta en `CHANGELOG.md`: versión, cambio principal, tecnologías, archivo fuente que lo evidencia y qué queda pendiente de validar.
6. Mantener los diagramas en `docs/arquitectura/` y las tres etapas en `docs/historial/`.

### Exclusiones recomendadas antes de publicar

No publicar `config/users.json` (aunque los usuarios sean ficticios), contraseñas, datos personales, rutas internas, imágenes originales de clientes, modelos grandes ni directorios de entrenamiento con información de prueba no autorizada. Las credenciales de ejemplo deben ir en documentación ficticia, no en ficheros reales de acceso. Evitar subir también `__pycache__/`, `.venv/` y artefactos locales.

**Importante:** si se añade accidentalmente un archivo sensible y luego se borra, podría seguir existiendo en el historial de Git. La revisión debe hacerse **antes del primer commit**.

### Ejemplo de mensajes de versión (no implican que ya se hayan publicado)

- `v10 — Clasificación de imágenes completas con PyTorch`
- `v11 — Detección y conteo con Faster R-CNN`
- `v12 — Separación de datos, entrenamiento y pruebas`
- `v13 — Ampliación de aumentación y ruido visual`
- `v14 — Anotaciones multiclase y migración a YOLO`
- `v15 — Perfiles, roles y mediciones con ArUco`
- `v16 — Secuencias, cámara en vivo e integración ESP32`

**Pendiente:** recuperación de v4–v9 y contraste con sus archivos originales para completar la primera etapa.