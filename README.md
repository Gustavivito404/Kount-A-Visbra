# Kount-A / Visbra

## Evolución del software para inspección y conteo por visión artificial

Este repositorio documenta el desarrollo incremental de **Visbra**, software que integra adquisición, etiquetado, procesamiento visual, detección mediante YOLO, referencias ArUco, interfaz HMI y comunicación con el sistema embebido del prototipo **Kount-A**.

### Tres grandes etapas

1. **Desarrollo inicial:** base de aumento de datos, bounding boxes, parámetros por clase y entrenamiento de clasificación (v4–v10).
2. **Detección y entrenamiento:** detector multiobjeto, herramientas de etiquetado y transición a YOLO (v11–v14).
3. **Integración del sistema:** gestión de perfiles, ArUco, interfaces, cámara en vivo, comunicación ESP32 y secuencias (v15–v16).

### Versiones

Las versiones se identifican con etiquetas del tipo `v1.4.0`, `v2.11.0` y `v3.16.4`. **El primer número indica la etapa**, el segundo corresponde al número original de Visbra y el tercero distingue una revisión puntual documentada. Esta convención es propia del proyecto, no SemVer estándar.

Los commits y las etiquetas se reconstruyeron a partir de entregas ZIP y no representan fechas originales de programación. No se dispone de un paquete v7 y la designación v4 para la carpeta inicial es provisional.

### Ejecutar la última entrega

Consulta su `README.md` y `requirements.txt` antes de instalar. Se requiere Python y las bibliotecas de visión/IA correspondientes; para actuadores, el firmware ESP32 debe estar instalado aparte. Los conjuntos de datos, los perfiles, los modelos y las configuraciones personales no se incluyen por privacidad y tamaño.

### Publicación y autoría

El proyecto pertenece a su equipo desarrollador. La publicación pública no concede por sí misma permiso de redistribución de código ajeno. No se ha definido una licencia en este paquete de preparación.


### Documentación publicada

- [Tres avances de desarrollo](docs/historial/AVANCES.md)
- [Registro de cambios](docs/historial/CHANGELOG.md)
- [Análisis inicial de las versiones](docs/historial/ANALISIS_INICIAL.md)
- [Texto para el capítulo 3 en LaTeX](docs/capitulo3/TEXTO_TESINA_CAPITULO_3.tex)
- [Guía para importar el historial Git y sus etiquetas](docs/historial/PUBLICACION.md)

**Pendiente:** transferir el historial Git completo y los archivos de diagramas vectoriales que están en el paquete preparado.
