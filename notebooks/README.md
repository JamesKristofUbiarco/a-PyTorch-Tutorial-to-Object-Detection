# Probar SSD300 con marimo

Desde la raíz del proyecto:

```bash
uv sync --extra notebook
uv run --extra notebook marimo edit notebooks/deteccion.py
```

Para abrir únicamente la aplicación, sin el editor:

```bash
uv run --extra notebook marimo run notebooks/deteccion.py
```

1. En el navegador, pulsa **Cargar checkpoint**. Por defecto usa el archivo
   `checkpoint_ssd300.pth.tar` de la raíz. Puedes indicar otra ruta, por ejemplo
   `saved_models/checkpoint_ssd300.pth.tar`. Usa checkpoints locales de este proyecto.
2. Selecciona **Subir imagen** y arrastra una foto, o selecciona **Cámara** y permite
   su acceso en el navegador. El widget ofrece captura manual y automática cada
   dos segundos. Detén la captura automática cuando termines.
3. Ajusta la confianza mínima, el solapamiento máximo para eliminar cajas
   duplicadas (NMS) y el número máximo de detecciones. Compara la imagen original
   con la anotada y consulta la tabla de clases, puntuaciones y coordenadas.

`auto` elige CUDA, MPS o CPU según disponibilidad. Si un dispositivo falla,
selecciona `cpu` y vuelve a cargar el checkpoint. La supresión de cajas se calcula
en CPU. No hace falta la fuente `calibril.ttf` de `detect.py`.

El modelo se carga al enviar el formulario; cambiar la entrada o los umbrales no
lo recarga. Para leer una versión nueva del checkpoint, vuelve a enviar el
formulario. El notebook no entrena ni modifica los checkpoints.

La inferencia se realiza en el proceso Python local, sin enviar fotos a un servicio
de detección. Marimo puede conservar resultados en su caché de sesión `__marimo__/`,
excluida de Git en esta carpeta.
La cámara requiere permisos del navegador; utiliza la dirección `localhost` que
abre marimo. La captura automática es periódica, no un flujo de video de alta
frecuencia. No es posible verificar el acceso a tu cámara sin tu navegador.

El modelo reconoce las 20 clases de Pascal VOC. Las puntuaciones permiten explorar
sus predicciones, pero no miden precisión ni mAP sin anotaciones de referencia.

Implementación de los widgets:
[WebcamCapture](https://koaning.github.io/wigglystuff/reference/webcam-capture/) y
[carga de archivos de marimo](https://docs.marimo.io/api/inputs/file/).
