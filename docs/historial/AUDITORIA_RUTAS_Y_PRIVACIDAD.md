# Auditoría de rutas locales y privacidad

En la reconstrucción documental se excluyeron las configuraciones privadas, y las rutas absolutas que identificaban equipos concretos se sustituyeron por `[RUTA_LOCAL_OMITIDA]`.

Las rutas relativas de código, como `src/gui_app.py` y `assets/kounta_logo.png`, se conservaron porque son necesarias y no identifican ninguna computadora.

La auditoría incluyó las 19 etiquetas y todos los objetos del historial Git.

**Rutas locales sustituidas en los archivos fuente:** 4.

**Nombres de red Wi-Fi sustituidos:** 3.

La dirección privada de la ESP32 `192.168.4.1`, el loopback `127.0.0.1` y el puerto `8080` permanecen porque son valores técnicos de conexión, no direcciones IP públicas ni credenciales.
