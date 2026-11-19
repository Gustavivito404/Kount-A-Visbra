# Publicación del historial Git completo

El historial original de desarrollo se reconstruyó localmente en un archivo `visbra_historial_completo.bundle`. Contiene 20 commits y 19 etiquetas, además de un estado final con documentación y código fuente.

**Estado:** documentación inicial ya disponible en GitHub, importación del historial completo pendiente.

## Cómo importar desde el paquete preparado

El archivo `KountA_Visbra_historial_GitHub_preparado.zip` contiene `visbra_historial_completo.bundle` y 19 instantáneas del código fuente. El archivo Git es el medio preferido para conservar commits y etiquetas.

Para evitar sobrescribir la rama `main` ya creada, se recomienda importar el historial a una rama distinta llamada `historial-visbra` y sus etiquetas. Esto permite verificar primero el código y después decidir cómo organizar la rama principal.

Ejemplo desde una computadora con Git instalado, tras extraer el paquete:

```bash
git clone https://github.com/Gustavivito404/Kount-A-Visbra.git
cd Kount-A-Visbra
git fetch ../visbra_historial_completo.bundle 'refs/heads/main:refs/remotes/importado/main' 'refs/tags/*:refs/tags/*'
git push origin refs/remotes/importado/main:refs/heads/historial-visbra
git push origin --tags
```

La ruta relativa del archivo `.bundle` depende de dónde hayas extraído el ZIP. Antes de ejecutar, comprueba que el archivo exista en esa ubicación.

No uses `--force` sobre la rama principal, pues tiene commits actuales. Al concluir, compara `historial-visbra` con `main` y decide si realizar una integración.

La reconstrucción elimina rutas de equipos personales y archivos de credenciales detectados, pero conviene revisar cualquier repositorio público antes de publicar material confidencial.
