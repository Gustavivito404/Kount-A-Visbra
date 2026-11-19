# Kount-A — Entrega de desarrollo v16

Sistema de inspección y conteo por visión artificial, anotación BBox, entrenamiento YOLO, medición ArUco y secuencias programables para la mesa vibratoria VISBRA. En la interfaz se muestra simplemente **Kount-A**, sin número de versión.

## 1. Instalación y arranque en Windows

Descomprime el proyecto en otra carpeta; **no reemplaces la v15** hasta verificar que todo funcione. Desde la carpeta que contiene `src`:

```bat
py -m pip install -r requirements.txt
py -m src.gui_app
```

Si utilizas un entorno virtual, actívalo antes. Si `python` apunta al Python de MSYS2 en vez de Windows, ejecuta el Python correcto de Windows o `.venv\Scripts\python.exe`. Para iniciar el programa utiliza **`python -m src.gui_app`**, no `python src/gui_app.py`.

El programa utiliza `opencv-contrib-python` para ArUco, y Ultralytics + PyTorch para YOLO. En Windows se instala también `pygrabber` para intentar obtener los nombres reales de cámaras mediante DirectShow. Si el controlador o Windows no permite obtenerlos, el selector mostrará índices y resolución. Los índices pueden variar al reconectar webcams; comprueba la vista previa.

## 2. Perfiles, usuarios y datos existentes

Se conserva la estructura de la v15 y los archivos del ZIP original: `assets/kounta_logo.png`, `config/users.json`, `config/appearance.json`, `config/ui_session_config.json` y `config/profiles_location.json`.

- **Usuario dueño:** administra otras cuentas; puede entrar a vista de ingeniero u operador.
- **Ingeniero:** crea y etiqueta perfiles, entrena modelos, configura ArUco, prueba conteos y modifica secuencias.
- **Operador:** usa únicamente perfiles/modelos autorizados y ejecuta la secuencia guardada por ingeniería.
- Se conservan las cuentas de prueba del ZIP recibido. Las contraseñas están almacenadas como derivaciones criptográficas, no texto plano. Protege `config/users.json` y **cambia esas contraseñas si dejas de usar la instalación como prueba**.
- `config/profiles_location.json` conserva la ubicación elegida de perfiles externos. Si la carpeta aún existe en la misma ruta Windows, se recuperará; si cambió, usa **Explorar perfiles** para seleccionarla. No se copian ni alteran automáticamente imágenes/modelos existentes.
- Las nuevas secuencias se guardan como `work_sequence.json` *dentro de la carpeta de cada perfil*. No se almacenan en la carpeta global de configuración.

## 3. Interfaz y etiquetado

Continúan disponibles las funciones de la v15: perfiles, BBox manual multiclase, propuesta por escala de grises, preetiquetado YOLO, aumento de imágenes y revisión, exportación del dataset, entrenamiento y administración de usuarios.

En BBox se conservan **Mostrar todas**, **Ocultar todas**, **Solo esta** y la columna **Ver**. Las cajas ocultas siguen almacenadas; si agregas otra caja con las anteriores ocultas, la nueva permanece visible.

La barra superior ofrece **Escala (90–150 %)**, tema claro/oscuro y un botón siempre visible para **Detener secuencia**. La columna de controles en **Prueba y conteo** puede desplazarse con la rueda del mouse. Las imágenes se ajustan al espacio disponible.

## 4. Cámara en vivo y YOLO

En **5 · Prueba y conteo**:

1. Pulsa **Buscar cámaras**. Elige el dispositivo; en Windows se muestra el nombre cuando DirectShow lo proporciona.
2. Selecciona tu modelo `best.pt` en vista de ingeniero; en operador se utiliza el modelo autorizado del perfil.
3. Pulsa **Iniciar cámara en vivo** para mostrar video con detecciones YOLO continuas. El modelo se carga una sola vez por sesión de video y se procesan los frames más recientes disponibles, sin acumular una cola de imágenes atrasadas.
4. **Detener cámara** conserva el último fotograma en pantalla. En ingeniería, **Contar objetos** también permite tomar el último fotograma, detener el video y obtener un conteo definitivo con los parámetros de ArUco del perfil.
5. **Guardar distribución** guarda la imagen *sin BBox dibujadas* en `<perfil>/input/mezclas/distribucion_FECHA.png`. Debes revisar y etiquetar esa imagen antes de incluirla en un nuevo entrenamiento.

No sumes conteos de fotogramas consecutivos; el número mostrado en directo es el conteo de un fotograma, no de toda la transmisión. La rapidez de YOLO depende de la GPU/CPU, resolución, modelo y carga del equipo.

### Calibración ArUco

En **6 · Calibración ArUco**, la webcam puede mostrarse en vivo con marcadores detectados, validación y error RMSE. Edita las medidas de referencia y observa la previsualización; **Guardar calibración del perfil** escribe los parámetros permanentes. Los marcadores usados por el módulo existente son ID 0 (superior izquierdo), 1 (superior derecho), 2 (inferior derecho) y 3 (inferior izquierdo).

Los modos `none`, `optional` y `required` se conservan. En el modo `required`, una calibración inválida impide la inspección definitiva y detiene la secuencia. En el modo `optional`, la inferencia puede continuar a foto completa si no se valida ArUco.

## 5. Editor de secuencias

El ingeniero dispone de **7 · Secuencias de trabajo**:

- **Agregar** crea un paso nuevo (después de la fila seleccionada, o al final).
- **Editar** abre el mismo formulario con los valores de la fila seleccionada. También funciona con doble clic.
- **Subir/Bajar** cambia el orden de ejecución.
- **Eliminar** quita una fila de la lista, previa confirmación.
- **Guardar secuencia** escribe `work_sequence.json` dentro del perfil.
- **Probar secuencia** ejecuta los pasos guardados y abre la pestaña de inspección. **Detener ejecución** intenta detener inmediatamente la operación.

La tabla principal es compacta y tiene desplazamiento vertical, sin límite visual de filas (máximo de validación: 500 instrucciones). Las ventanas de edición son dinámicas:

| Instrucción | Configuración |
|---|---|
| **Movimiento de motor** | Motor 1 o 2, sentido horario/antihorario, PWM 0–100%, tiempo 0.1–120 s |
| **LED** | Encender/apagar; RGB rojo, verde y azul de 0 a 255; selector de color |
| **Cámara** | Automática o manual; cámara asignada (índice/nombre); espera de estabilización |

**Modo automático:** se espera la estabilización, se captura una foto, se ejecuta un conteo y se avanza a la instrucción siguiente.

**Modo manual:** aparece video con detecciones YOLO continuamente. El operador puede intervenir físicamente en el lote con los motores detenidos. **CONFIRMAR Y CONTINUAR** realiza el conteo definitivo, libera la cámara y continúa con la instrucción siguiente.

En cada paso de cámara se selecciona el dispositivo: se pueden utilizar distintas webcams de una misma computadora. Si se cambian las conexiones USB, vuelve a buscar cámaras y verifica que cada secuencia apunte al dispositivo correcto.

## 6. Control de ESP32 sin reprogramar firmware

El cliente `src/esp32_link.py` reutiliza exactamente el protocolo de tu compañero:

```text
M_RUN:dir_m1:dir_m2:pwm_m1:pwm_m2:tiempo_ms
RGB:r:g:b
MOTOR:0
```

La ESP32 actúa como punto de acceso Wi-Fi; conecta la computadora a **[RED_WIFI_ESP32]**, utilizando la contraseña de tu firmware, y verifica que la IP `192.168.4.1` y puerto `8080` correspondan a tu equipo. En ingeniería puedes cambiarlos en el panel de Prueba y conteo; se conservan en la configuración de sesión. El operador utiliza los valores guardados.

**Importante:** el firmware responde `OK` cuando acepta la instrucción, **no cuando el movimiento ha terminado**. El ejecutor Python espera el tiempo programado más un margen conservador para rampas y frenado y envía una orden de parada después de cada movimiento. Esta espera es estimativa, no una confirmación de fin obtenida de la ESP32. Se deberá verificar el tiempo real de detención mecánica.

El software no modifica el firmware que ya subiste. **Sigue pendiente la mejora de seguridad en el ESP32**: watchdog ante pérdida de comunicación, recepción TCP con timeout y confirmación del estado final de los motores. Una parada enviada por Wi-Fi **no reemplaza un paro físico de emergencia**. No operes pruebas desatendidas ni cerca de piezas móviles sin protecciones adecuadas.

## 7. Pruebas incluidas y límites

```bat
python -m src.smoke_test
python -m src.smoke_v16
```

- `smoke_test` verifica los procesos heredados de anotación, aumentación, exportación y partición de dataset.
- `smoke_v16` verifica formato y guardado de secuencias, validación de parámetros y comunicación con una **ESP32 simulada por TCP local**.
- Durante la preparación se comprobó la apertura y navegación de las ocho pestañas, editor y guardado de fotos, secuencias de LED/motor simuladas, modos de cámara simulados, y cambio real de tamaño de los controles.

**No se han validado en este entorno**: webcams físicas de tu computadora, ejecución de un modelo YOLO real con tus pesos, recepción del firmware en tu ESP32 ni el comportamiento mecánico de los motores. Antes de una operación real verifica el sistema por etapas: cámaras, LED, movimientos individuales breves, paro, secuencia completa y conteo con piezas conocidas.

Recomendación de instalación: conserva una copia íntegra de tu v15 y una copia de tu **carpeta externa de perfiles** antes de probar la v16. Nunca experimentes sobre los únicos originales de un dataset o modelo.
