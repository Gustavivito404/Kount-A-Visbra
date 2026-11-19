# Evolución de Visbra para Kount-A

Este documento identifica tres avances principales del desarrollo. Los números son una convención **propia**: `v<etapa>.<versión original>.<revisión>`.

## Etapa 1 · Desarrollo inicial (v4 a v10)

Se implementaron herramientas para preparar imágenes, gestionar cuadros delimitadores y umbrales por clase, junto con una primera aproximación al entrenamiento y clasificación.

Etiquetas publicadas: `v1.4.0`, `v1.5.0`, `v1.6.0`, `v1.8.0`, `v1.9.0`, `v1.10.0`. **No se dispone de entrega completa independiente de v7**. La numeración inicial v4 es provisional.

## Etapa 2 · Detección y entrenamiento (v11 a v14)

Se incorporaron detección de múltiples objetos, pruebas con distintas muestras, mejoras de aumento de datos y un flujo de etiquetado y entrenamiento dirigido hacia YOLO.

Etiquetas publicadas: `v2.11.0`, `v2.12.0`, `v2.13.0`, `v2.14.0`.

## Etapa 3 · Integración del sistema (v15 a v16)

Se integraron perfiles, roles, ArUco, HMI, cámara en vivo y comunicación con el microcontrolador ESP32. Posteriormente se desarrollaron secuencias de trabajo, estimación de saturación visual y mejoras de usabilidad.

Etiquetas publicadas: `v3.15.0`–`v3.15.3` y `v3.16.0`–`v3.16.4`.

### Nota de trazabilidad

Los archivos ZIP originales permitieron reconstruir un historial cronológico de 20 commits y 19 etiquetas. **Ese historial ya se encuentra publicado en la rama `historial-visbra` y en sus 19 etiquetas**. No se deben asignar fechas originales a commits reconstruidos; las fechas reales se reservan a su documentación de origen.
