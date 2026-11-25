"""Interfaz Streamlit para SSD300: cámara en vivo, foto de cámara o imagen subida.

Uso (desde la raíz del repo):
    pip install streamlit streamlit-webrtc opencv-python-headless
    streamlit run app_streamlit.py

Reutiliza la clase `Detector` de notebooks/inference.py (sin tocar el código original).
El tema base está en .streamlit/config.toml; el resto del estilo está en la constante CSS.
"""

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "notebooks"))
from inference import Detector, read_image  # noqa: E402
from utils import label_color_map, voc_labels  # noqa: E402  (la raíz queda en sys.path vía inference)

DEFAULT_CKPT = "checkpoint_ssd300.pth.tar"

st.set_page_config(page_title="SSD300 · Detección de objetos", page_icon="🎯", layout="wide")

# ---------- Estilo ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

/* Fondo y contenedor */
.stApp {
  background: radial-gradient(1200px 600px at 85% -10%, rgba(124,131,255,.16), transparent 60%),
              radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,.10), transparent 55%),
              #0B1020;
}
.block-container { padding-top: 2rem; max-width: 1280px; }
footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }

/* Hero */
.hero {
  display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap;
  padding: 1.8rem 2rem; margin-bottom: 1.4rem; border-radius: 20px;
  background: linear-gradient(135deg, rgba(124,131,255,.22), rgba(34,211,238,.10) 60%, rgba(18,26,51,.6));
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 10px 40px rgba(0,0,0,.35);
}
.hero .eyebrow { font-size: .75rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: #9AA4FF; }
.hero h1 { margin: .25rem 0 .35rem; padding: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, #fff, #B8BEFF); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { margin: 0; color: #A8B0D0; font-size: .98rem; max-width: 560px; }
.chips { display: flex; gap: .5rem; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: center; gap: .45rem; padding: .38rem .8rem; border-radius: 999px;
  font-size: .78rem; font-weight: 500; color: #D5DAF5; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #34D399; box-shadow: 0 0 10px #34D399; }

/* Tabs */
[role="tablist"] { gap: .4rem; background: rgba(255,255,255,.05); padding: .35rem; border-radius: 14px; width: fit-content; border: 1px solid rgba(255,255,255,.06); }
[data-testid="stTab"], button[role="tab"] { height: 42px; padding: 0 1.2rem !important; border-radius: 10px !important; color: #A8B0D0; background: transparent; }
[data-testid="stTab"] p, button[role="tab"] p { font-weight: 600; }
[data-testid="stTab"][aria-selected="true"], button[role="tab"][aria-selected="true"] { background: linear-gradient(135deg, #7C83FF, #5B63F0) !important; color: #fff !important; box-shadow: 0 4px 14px rgba(124,131,255,.45); }
[data-testid="stTab"][aria-selected="true"] p { color: #fff !important; }
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] { padding-top: 1rem; }

/* Tarjetas de métricas */
.m-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .9rem; margin: 1rem 0 1.2rem; }
.m-card { padding: 1rem 1.2rem; border-radius: 16px; background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08); }
.m-label { font-size: .72rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: #8D96BC; }
.m-value { font-size: 1.9rem; font-weight: 800; color: #fff; letter-spacing: -.02em; line-height: 1.2; }
.m-unit { font-size: .85rem; font-weight: 500; color: #8D96BC; margin-left: .2rem; }

/* Paneles */
.panel { padding: 1rem 1.2rem; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.07); margin-bottom: 1rem; color: #B9C0DE; font-size: .92rem; }
.panel b { color: #fff; }
.legend { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .6rem; }
.tag { padding: .22rem .65rem; border-radius: 999px; font-size: .75rem; font-weight: 600; color: #fff; }
.cap { font-size: .8rem; font-weight: 600; color: #8D96BC; letter-spacing: .06em; text-transform: uppercase; margin: .2rem 0 .5rem; }

/* Imágenes, video y componentes */
[data-testid="stImage"] img { border-radius: 16px; border: 1px solid rgba(255,255,255,.08); box-shadow: 0 10px 30px rgba(0,0,0,.35); }
iframe { border-radius: 16px; }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed rgba(124,131,255,.45); border-radius: 16px; background: rgba(124,131,255,.06); }
[data-testid="stCameraInput"] > div, [data-testid="stCameraInputWebcamStyledBox"] { border-radius: 16px; overflow: hidden; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,.08); }
.stButton > button, [data-testid="stBaseButton-secondary"] { border-radius: 10px; font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0E1530, #0A0F22); border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] h2 { font-size: 1.1rem; font-weight: 700; }
.side-title { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #8D96BC; margin: 1.1rem 0 .2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- Modelo ----------
@st.cache_resource(show_spinner="Cargando checkpoint...")
def load_detector(path: str, device: str) -> Detector:
    return Detector(path, device)


def hex_to_bgr(color: str):
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_rows(frame_bgr: np.ndarray, rows: list) -> np.ndarray:
    """Dibuja las detecciones (rows de Detector.predict) sobre un frame BGR."""
    out = frame_bgr.copy()
    for r in rows:
        color = hex_to_bgr(label_color_map[r["Clase"]])
        p1, p2 = (r["x1"], r["y1"]), (r["x2"], r["y2"])
        cv2.rectangle(out, p1, p2, color, 3)
        text = f'{r["Clase"]} {r["Confianza (%)"]:.0f}%'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(r["y1"], th + 8)
        cv2.rectangle(out, (r["x1"], y0 - th - 8), (r["x1"] + tw + 6, y0), color, -1)
        cv2.putText(out, text, (r["x1"] + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------- Helpers de UI ----------
def metric_cards(items):
    """items: lista de (etiqueta, valor, unidad)."""
    cards = "".join(
        f'<div class="m-card"><div class="m-label">{l}</div>'
        f'<div class="m-value">{v}<span class="m-unit">{u}</span></div></div>' for l, v, u in items)
    st.markdown(f'<div class="m-row">{cards}</div>', unsafe_allow_html=True)


def class_legend():
    tags = "".join(f'<span class="tag" style="background:{label_color_map[c]}">{c}</span>' for c in voc_labels)
    return f'<div class="legend">{tags}</div>'


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown('<div class="side-title">Modelo</div>', unsafe_allow_html=True)
    ckpt = st.text_input("Ruta del checkpoint", DEFAULT_CKPT)
    device = st.selectbox("Dispositivo", ["auto", "cpu", "cuda", "mps"])
    st.markdown('<div class="side-title">Detección</div>', unsafe_allow_html=True)
    min_score = st.slider("Confianza mínima", 0.05, 0.95, 0.40, 0.05,
                          help="Umbral mínimo para aceptar una detección.")
    max_overlap = st.slider("Solapamiento máx. (NMS)", 0.1, 0.9, 0.45, 0.05,
                            help="Cajas más solapadas que esto se consideran duplicadas.")
    top_k = st.slider("Máx. detecciones", 1, 100, 50)

try:
    detector = load_detector(ckpt, device)
except Exception as e:  # checkpoint ausente/corrupto
    st.error(f"No se pudo cargar el checkpoint: {e}")
    st.stop()

# ---------- Encabezado ----------
st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="eyebrow">SSD300 · PyTorch</div>
        <h1>Detección de objetos</h1>
        <p>Identifica 20 clases de objetos en tiempo real con tu cámara, o analiza fotos e imágenes.</p>
      </div>
      <div class="chips">
        <span class="chip"><span class="dot"></span>Modelo cargado</span>
        <span class="chip">📦 {detector.path.name}</span>
        <span class="chip">🔁 Época {detector.epoch}</span>
        <span class="chip">🖥️ {str(detector.device).upper()}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_live, tab_photo, tab_upload = st.tabs(["🎥  Cámara en vivo", "📸  Foto con cámara", "🖼️  Subir imagen"])


def show_result(image: Image.Image):
    annotated, rows, secs = detector.predict(image, min_score, max_overlap, top_k)
    best = max((r["Confianza (%)"] for r in rows), default=0)
    metric_cards([
        ("Detecciones", len(rows), ""),
        ("Clases distintas", len({r["Clase"] for r in rows}), ""),
        ("Mayor confianza", f"{best:.0f}", "%"),
        ("Latencia", f"{secs * 1000:.0f}", "ms"),
    ])
    c1, c2 = st.columns(2)
    c1.markdown('<div class="cap">Original</div>', unsafe_allow_html=True)
    c1.image(image, use_container_width=True)
    c2.markdown('<div class="cap">Detecciones</div>', unsafe_allow_html=True)
    c2.image(annotated, use_container_width=True)
    if rows:
        st.markdown('<div class="cap">Detalle</div>', unsafe_allow_html=True)
        st.dataframe(
            rows, use_container_width=True, hide_index=True,
            column_config={
                "Confianza (%)": st.column_config.ProgressColumn(
                    "Confianza", min_value=0, max_value=100, format="%.1f%%"),
            },
        )
    else:
        st.info("Sin detecciones; baja la confianza mínima en la barra lateral.")


# ---------- Cámara en vivo (streamlit-webrtc) ----------
with tab_live:
    try:
        import av
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        st.warning("Instala `streamlit-webrtc` para usar la cámara en vivo.")
    else:
        class Processor:
            """Inferencia en un hilo aparte; el video se dibuja con las últimas cajas."""

            def __init__(self):
                self.lock = threading.Lock()
                self.latest = None      # último frame RGB pendiente de inferir
                self.rows = []
                self.params = (0.4, 0.45, 50)
                self.stop = threading.Event()
                self.ev = threading.Event()
                threading.Thread(target=self._worker, daemon=True).start()

            def _worker(self):
                while not self.stop.is_set():
                    self.ev.wait(timeout=0.5)
                    self.ev.clear()
                    with self.lock:
                        frame, params = self.latest, self.params
                        self.latest = None
                    if frame is None:
                        continue
                    _, rows, _ = detector.predict(Image.fromarray(frame), *params)
                    with self.lock:
                        self.rows = rows

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                with self.lock:
                    self.latest = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.params = (min_score, max_overlap, top_k)
                    rows = list(self.rows)
                self.ev.set()
                return av.VideoFrame.from_ndarray(draw_rows(img, rows), format="bgr24")

        st.markdown('<div class="panel">Pulsa <b>START</b> y permite el acceso a la cámara. '
                    'Ajusta la confianza y el NMS desde la barra lateral mientras corre.</div>',
                    unsafe_allow_html=True)
        webrtc_streamer(
            key="ssd-live",
            video_processor_factory=Processor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        with st.expander("Clases que puede detectar"):
            st.markdown(class_legend(), unsafe_allow_html=True)

# ---------- Foto con cámara ----------
with tab_photo:
    shot = st.camera_input("Toma una foto", label_visibility="collapsed")
    if shot is not None:
        show_result(read_image(shot.getvalue()))

# ---------- Subir imagen ----------
with tab_upload:
    up = st.file_uploader("Arrastra una imagen o selecciónala", type=["jpg", "jpeg", "png"])
    if up is not None:
        show_result(read_image(up.getvalue()))
