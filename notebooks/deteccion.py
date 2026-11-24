import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="SSD300 · Cámara e imágenes")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    from PIL import UnidentifiedImageError
    from wigglystuff import WebcamCapture

    notebook_dir = Path(__file__).resolve().parent
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    from inference import Detector, read_image
    from utils import voc_labels

    return (
        Detector,
        UnidentifiedImageError,
        WebcamCapture,
        mo,
        read_image,
        voc_labels,
    )


@app.cell
def _(mo):
    mo.md("""
    # Prueba tu detector SSD300

    Carga tu checkpoint y prueba una foto o una captura de cámara.
    Las cajas muestran la clase y la confianza de cada detección.
    """)
    return


@app.cell
def _(mo):
    checkpoint_form = mo.ui.dictionary({
        "ruta": mo.ui.text(value="checkpoint_ssd300.pth.tar", label="Ruta del checkpoint", full_width=True),
        "dispositivo": mo.ui.dropdown(
            options=["auto", "cpu", "mps", "cuda"], value="auto", label="Dispositivo"),
    }).form(submit_button_label="Cargar checkpoint")
    mo.vstack([
        mo.md("### 1. Modelo\nLa ruta es relativa a la raíz del proyecto. También puedes usar `saved_models/checkpoint_ssd300.pth.tar`."),
        checkpoint_form,
    ])
    return (checkpoint_form,)


@app.cell
def _(Detector, checkpoint_form, mo):
    mo.stop(checkpoint_form.value is None, mo.md("Pulsa **Cargar checkpoint** para comenzar."))
    detector = None
    _load_error = None
    try:
        detector = Detector(checkpoint_form.value["ruta"], checkpoint_form.value["dispositivo"])
    except Exception as _exc:
        _load_error = str(_exc)
    mo.stop(_load_error is not None, mo.callout(f"No se pudo cargar el modelo: {_load_error}", kind="danger"))
    mo.callout(
        f"Modelo cargado · dispositivo: **{detector.device}** · "
        f"época guardada: **{detector.epoch}** · pasos: **{detector.steps if detector.steps is not None else 'sin registro'}**",
        kind="success",
    )
    return (detector,)


@app.cell
def _(mo):
    source = mo.ui.radio(options=["Subir imagen", "Cámara"], value="Subir imagen", label="Entrada")
    confidence = mo.ui.slider(start=0.05, stop=0.95, step=0.05, value=0.4, label="Confianza mínima", show_value=True)
    overlap = mo.ui.slider(start=0.1, stop=0.9, step=0.05, value=0.45, label="Solapamiento máximo (NMS)", show_value=True)
    top_k = mo.ui.slider(start=1, stop=100, step=1, value=50, label="Máximo de detecciones", show_value=True)
    mo.vstack([
        mo.md("### 2. Entrada y ajustes"), source,
        mo.hstack([confidence, overlap, top_k], wrap=True),
    ])
    return confidence, overlap, source, top_k


@app.cell
def _(WebcamCapture, mo, source):
    camera = None
    upload = None
    if source.value == "Cámara":
        camera = mo.ui.anywidget(WebcamCapture(interval_ms=2000))
        _input_view = mo.vstack([
            mo.md("Permite el acceso a la cámara. Usa **Capture** para una foto o activa la captura automática para actualizar cada 2 segundos."),
            camera,
        ])
    else:
        upload = mo.ui.file(filetypes=["image/*"], multiple=False, kind="area", label="Selecciona o arrastra una imagen")
        _input_view = upload
    _input_view
    return camera, upload


@app.cell
def _(UnidentifiedImageError, camera, mo, read_image, source, upload):
    _contents = None
    if source.value == "Cámara" and camera is not None:
        mo.stop(bool(camera.error), mo.callout(f"No se pudo abrir la cámara: {camera.error}. Revisa los permisos del navegador o sube una imagen.", kind="warn"))
        # Reading a synced trait makes each capture reactive in marimo.
        if camera.image_base64:
            _contents = camera.get_bytes()
    elif upload is not None:
        _contents = upload.contents()
    mo.stop(not _contents, mo.md("Sube una imagen o captura una foto para ver las detecciones."))
    original = None
    _image_error = None
    try:
        original = read_image(_contents)
    except (UnidentifiedImageError, OSError, ValueError) as _exc:
        _image_error = str(_exc)
    mo.stop(_image_error is not None, mo.callout(f"No se pudo leer la imagen: {_image_error}", kind="danger"))
    return (original,)


@app.cell
def _(confidence, detector, mo, original, overlap, top_k):
    _inference_error = None
    annotated, detections, elapsed = None, [], 0.0
    try:
        annotated, detections, elapsed = detector.predict(
            original, min_score=confidence.value, max_overlap=overlap.value, top_k=top_k.value)
    except Exception as _exc:
        _inference_error = str(_exc)
    mo.stop(_inference_error is not None, mo.callout(f"No se pudo ejecutar la detección: {_inference_error}. Si falla MPS, carga el modelo en CPU.", kind="danger"))
    return annotated, detections, elapsed


@app.cell
def _(annotated, detections, elapsed, mo, original):
    mo.vstack([
        mo.md(f"### 3. Resultado\n**{len(detections)} detecciones** · inferencia y anotación: **{elapsed * 1000:.0f} ms**"),
        mo.hstack([
            mo.vstack([mo.md("**Original**"), mo.image(original, width=560)]),
            mo.vstack([mo.md("**Detecciones**"), mo.image(annotated, width=560)]),
        ], wrap=True),
        mo.ui.table(detections, selection=None, label="Objetos detectados") if detections else
        mo.callout("No se encontraron objetos con este umbral. Prueba bajar la confianza mínima o cambiar la imagen.", kind="info"),
    ])
    return


@app.cell
def _(mo, voc_labels):
    mo.accordion({
        "Clases y cómo interpretar el resultado": mo.md(
            "**Clases de Pascal VOC:** " + ", ".join(voc_labels) + ".\n\n"
            "La confianza es una puntuación del modelo; para medir su precisión se necesitan anotaciones de referencia. "
            "Este notebook sirve para una revisión visual. La captura automática procesa fotos periódicas, no video a 30 FPS."
        ),
    })
    return


if __name__ == "__main__":
    app.run()
