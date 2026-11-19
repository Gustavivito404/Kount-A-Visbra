# Visbra / Kount-A — v1.4.0

**Reconstrucción documental de la entrega histórica de Visbra.**

- **Etiqueta técnica:** `v1.4.0`
- **Etapa:** 1 de 3
- **Versión original del proyecto:** 4
- **Revisión documentada:** 0
- **Cambio principal:** Primer entorno gráfico de aumento de imágenes y BBox sobre fondo claro.
- **Procedencia:** `augment_gui_project`
- **Nota:** Entrega base sin numeración explícita: asignación v4 inferida por orden y funciones frente a v5.

La numeración de GitHub es una convención documental personalizada, **no implica compatibilidad con versionado semántico estándar**. Los commits y las etiquetas fueron reconstruidos a partir de archivos proporcionados y **no representan fechas originales de desarrollo**.

## Ejecución

Consultar la documentación original (cuando está incluida) en `docs/README_ORIGINAL.md`. En general, desde la carpeta del proyecto:

```bash
python -m pip install -r requirements.txt
python -m src.gui_app
```

**Excepción:** Las primeras entregas pueden requerir `python src/gui_app.py`, y los requisitos o carpetas pueden no encontrarse en exactamente el mismo estado. Consulta el README original.

## Antes de utilizarlo

Los materiales personales del archivo histórico se excluyeron expresamente: usuarios, hashes de contraseñas, rutas locales, sesiones, datasets, imágenes de prueba, salidas y pesos de entrenamiento. Este código puede requerir volver a configurar el entorno, instalar dependencias y preparar un dataset propio para operar.

La publicación pública del repositorio **no equivale a otorgar una licencia de reutilización**. El equipo deberá decidir una licencia cuando corresponda.
