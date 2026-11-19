# Kount-A — v15 (perfiles, roles, temas, YOLO y ArUco)

## Cómo instalar y abrir (Windows)

Instala Python de Windows (se recomienda 3.11 o 3.12, 64 bits) y ejecuta **desde la carpeta que contiene `src`**:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.gui_app
```

Si tu Python Windows tiene otra versión, sustituye `py -3.11` por el ejecutable correspondiente. Si `python` apunta a `[RUTA_LOCAL_OMITIDA]`, utiliza el Python de Windows o el ejecutable de `.venv\Scripts\python.exe`. **No ejecutes `python src/gui_app.py`**; usa `python -m src.gui_app`.

Para equipos **sin NVIDIA**: instala PyTorch/torchvision CPU en el entorno virtual:

```bat
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

ArUco necesita `cv2.aruco` y se incluye `opencv-contrib-python`. Si la instalación existente de OpenCV presenta conflictos, desinstala `opencv-python` y `opencv-contrib-python` en **el mismo entorno** y reinstala únicamente `opencv-contrib-python`, comprobando que Ultralytics siga funcionando.

## Inicio de sesión y apariencia

- En la **primera ejecución**, el programa te pedirá crear una cuenta de **ingeniero** con usuario y contraseña (mínimo 8 caracteres).
- El ingeniero puede crear cuentas de `operator` o `engineer` desde la pestaña **Perfiles**.
- No se incluyen cuentas de demostración ni contraseñas predeterminadas. Las contraseñas se almacenan con PBKDF2-HMAC-SHA256 y sal individual en `config/users.json`. Mantén esa carpeta privada y protegida en Windows.
- El selector `☀ Claro / ☾ Oscuro` cambia fondo, textos, botones, pestañas y vistas existentes. La preferencia se guarda en `config/appearance.json`.
- `assets/kounta_logo.png` es un **logotipo tipográfico provisional**, no el logotipo oficial: puedes reemplazarlo por tu PNG definitivo conservando el nombre y la transparencia. Se presenta al iniciar sesión.

## Ingeniero / operador

**Ingeniero** dispone de las seis pestañas: Perfiles; BBox y etiquetado; Augmentación; Entrenamiento YOLO; Prueba y conteo; Calibración ArUco.

**Operador** solo tiene Producción: selecciona uno de los perfiles **autorizados** por ingeniería, cámara, resolución y enfoque (si lo admite el dispositivo), elige `Combinado` o `Comparativo` si ArUco está habilitado y pulsa **CAPTURAR Y ANALIZAR**. Los modelos de producción son bloqueados por el administrador del perfil.

## Perfiles y clases dinámicas

Un perfil creado desde el programa genera automáticamente:

```text
profiles/<perfil>/
  config.json
  input/<clase_1>/
  input/<clase_2>/
  input/mezclas/
  input/.kounta_annotations/       # aparece al etiquetar
  annotations/                    # reservada
  output/augmented_images/
  output/augmented_labels/
  output/yolo_dataset/            # aparece al exportar
  output/models/
  results/                        # reservado; no se registran inspecciones todavía
```

- Crear perfiles, agregar clases e importar fotos: **solo desde Perfiles**.
- Los nombres de clase se normalizan para YOLO (`Luneta fija` → `luneta_fija`). Se usan en el combo BBox, en las anotaciones, al aumentar imágenes y en el YAML de entrenamiento.
- `mezclas` es carpeta auxiliar, no clase. Permite anotar múltiples clases en una misma foto.
- Una clase nueva **invalida la autorización anterior**; hay que evaluar/autorizar una versión compatible. La aplicación impide eliminar clases cuando hay anotaciones o modelos que podrían depender de ellas.
- Modelos distintos por perfil en `output/models`. El ingeniero selecciona el `.pt` de esa carpeta, la aplicación comprueba su mapa de clases y lo autoriza para producción. Los modelos viejos permanecen guardados.
- **No cambies manualmente clases o carpetas** desde Windows: usa las funciones de la interfaz.

## Migración de datos desde v14

La v15 no copia ni elimina archivos de la v14 por su cuenta. Para recuperar anotaciones y entrenamientos previos:

1. Conserva tu proyecto v14 como copia de seguridad.
2. En v15, entra como ingeniero y selecciona el perfil predeterminado **Tornillería** (seis clases originales).
3. En **Perfiles**, elige **Importar proyecto v14 existente** y selecciona la carpeta que contenga `input` y `output` de v14.
4. Confirma la copia. Esta operación **copia** las carpetas completas con imágenes originales, JSON BBox, aumentaciones y modelos: puede requerir bastante espacio. Se exige un perfil de destino vacío y compatible.
5. Comprueba el dataset y el modelo importado. Si deseas utilizarlo con operador, selecciónalo en **Autorizar modelo**.

Los botones de v14 para **borrar aumentaciones** y **borrar exportación YOLO** siguen disponibles: solo afectan al `output` del perfil activo, no a imágenes originales, anotaciones o modelos. El segundo elimina también `split_assignments.json`, por lo que la siguiente exportación puede cambiar la partición de imágenes.

## Una operación de identificación y medición

Cada perfil configura ArUco como:

- `none`: YOLO cuenta y clasifica toda la foto, **sin ArUco ni dimensiones**.
- `optional`: intenta detectar los cuatro ArUco. Si son válidos, delimita zona y mide; de lo contrario, YOLO **cuenta toda la foto**, muestra la advertencia y no mide.
- `required`: sin cuatro ArUco y calibración válida, **bloquea la inspección**.

La prueba utiliza una sola fotografía. IDs de los marcadores: `0` superior izquierda, `1` superior derecha, `2` inferior derecha y `3` inferior izquierda. El ingeniero debe registrar las distancias **centro-a-centro** reales en mm y el tamaño físico del ArUco. La homografía convierte coordenadas de imagen al plano de referencia. Se comprueba también un RMSE aproximado entre tamaño de marcador calculado y tamaño conocido. Las distancias son **estimaciones planas**, no medidas 3D.

- `Combinado`: YOLO identifica y cuenta; las detecciones dentro de la zona válida se asocian aproximadamente a contornos OpenCV para reportar largo/ancho. Cuando falla el emparejamiento, las medidas se dejan en blanco.
- `Comparativo`: además muestra el **conteo independiente por contornos** dentro de la misma región y la diferencia respecto a YOLO. Si no hay calibración, la comparación se deshabilita, aunque YOLO sigue contando si ArUco es opcional.
- Contornos pueden fusionar piezas superpuestas y las medidas son preliminares: **valídalas con muestras y calibrador** antes de tratarlas como metrología industrial.

## Funciones conservadas de v14

BBox manual multiobjeto; ayuda por umbral gris; pre-etiquetado con YOLO; transformaciones torchvision v2 y revisión de aumentaciones; split persistido sin fuga de datos; exportación YOLO; fine-tuning, barras de entrenamiento, informes y `best.pt`; selección/búsqueda de webcams, captura, confianza, IoU NMS, CPU/CUDA; borrado seguro de datos generados.

## Limitaciones y pruebas

- El sistema inicia y se ha comprobado la navegación de ambas cuentas, los perfiles, los JSON BBox y el núcleo geométrico ArUco con imágenes sintéticas en Linux. **No se ha validado todavía la cámara física ni el entrenamiento/inferencia real de Ultralytics con tu dataset y hardware Windows.**
- El modo comparativo **no constituye una verificación de exactitud por sí mismo**; hace falta un conteo real etiquetado como referencia.
- No hay aún historial automático de inspecciones, exportación de producción ni control de actuadores; no se implementaron al no estar aprobados esos flujos.
- Evita mover manualmente los archivos aprobados; la autorización guarda la ruta del modelo dentro del perfil.
