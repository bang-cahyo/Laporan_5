# ======================================
# dashboard.py - YOLO Face Detection
# ======================================
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os
import cv2
import PIL.ImageOps

# ======================================
# Konfigurasi Halaman
# ======================================
st.set_page_config(
    page_title="YOLO Face Detection Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ======================================
# CSS Futuristik / Neon
# ======================================
st.markdown("""
<style>
body, .stApp { background: radial-gradient(circle at top left, #0b0f19, #0d1532, #121b3e); color: #eaeaea; font-family: 'Poppins', sans-serif; transition: all 0.3s ease; }
h1, h2, h3 { font-family: 'Poppins', sans-serif; font-weight: 700; }
h1 { font-size: 3rem; background: linear-gradient(90deg, #00e0ff, #7a00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
.stButton>button { background: linear-gradient(90deg, #00e0ff, #7a00ff); color: white; font-weight: 600; border: none; border-radius: 12px; padding: 0.7em 2em; margin: 5px; transition: all 0.3s ease; }
.stButton>button:hover { box-shadow: 0 0 25px #00e0ff80; transform: translateY(-2px); }
.result-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px); border-radius: 20px; padding: 20px; margin-top: 20px; transition: all 0.4s ease-in-out; }
.result-card:hover { box-shadow: 0 0 25px #00e0ff80, 0 0 50px #7a00ff60; transform: translateY(-5px); }
.neon-title { animation: glow 1.8s infinite alternate; font-weight: 800; font-size: 2.2rem; }
@keyframes glow { 0% { text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff; } 50% { text-shadow: 0 0 15px #00e0ff, 0 0 25px #7a00ff; } 100% { text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff; } }
footer { background: linear-gradient(90deg, #0d0f1a, #1e1f3a); padding: 15px 20px; border-radius: 12px; margin-top: 50px; text-align: center; font-size: 0.9rem; color: #bcd4ff; box-shadow: 0 0 20px #00e0ff40; }
</style>
""", unsafe_allow_html=True)

# ======================================
# Utility Functions
# ======================================
def get_downloadable_image(np_img):
    image = Image.fromarray(np_img)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def letterbox_image(img, target_size=(640,640)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - nh)//2
    left = (target_w - nw)//2
    canvas[top:top+nh, left:left+nw, :] = img_resized
    return canvas

# ======================================
# Tombol Back / Home
# ======================================
def go_home_button():
    if st.button("🏠 Kembali ke Home"):
        st.session_state.page = "home"
        st.experimental_rerun()

# ======================================
# Load YOLO Model
# ======================================
@st.cache_resource
def load_model():
    model_path = "model/Cahyo_Laporan4.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model {model_path} tidak ditemukan!")
        return None

model = load_model()

# ======================================
# Session State & Sidebar Navigasi
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "home"

with st.sidebar:
    st.markdown("<h3 style='color:#00e0ff; text-align:center;'>🤖 YOLO Face Detection</h3>", unsafe_allow_html=True)
    menu_options = {"🏠 Home": "home", "🧍 About": "about", "📷 Deteksi Wajah": "detect"}
    selected = st.radio("Navigasi:", list(menu_options.keys()), index=list(menu_options.values()).index(st.session_state.page))
    st.session_state.page = menu_options[selected]

# ======================================
# Halaman Home
# ======================================
def show_home():
    st.markdown("<h1 class='neon-title' style='text-align:center;'>🤖 YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#bcd4ff;'>Deteksi wajah & ekspresi secara real-time dengan YOLOv8.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Tombol navigasi cepat
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧍 Pelajari Tentang Aplikasi"):
            st.session_state.page = "about"
            st.experimental_rerun()
    with col2:
        if st.button("📷 Mulai Deteksi Wajah"):
            st.session_state.page = "detect"
            st.experimental_rerun()

    # Fitur singkat
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("<div class='result-card' style='text-align:center;'>📷 Upload & Kamera</div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='result-card' style='text-align:center;'>✅ Deteksi Otomatis</div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='result-card' style='text-align:center;'>💾 Simpan Hasil</div>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:#bcd4ff; margin-top:40px;'>Dikembangkan oleh <b>Heru Bagus Cahyo</b></p>", unsafe_allow_html=True)

# ======================================
# Halaman About
# ======================================
def show_about():
    go_home_button()  # Tombol kembali
    st.markdown('<h1 class="neon-title">About This App</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#bcd4ff;">Tentang aplikasi dan pembuatnya</p>', unsafe_allow_html=True)

    about_option = st.radio("Pilih:", ["Tentang Website", "Tentang Penulis"], index=0)
    if about_option == "Tentang Website":
        st.markdown("""
        Website ini dibuat untuk mendeteksi wajah menggunakan **YOLOv8**.
        **Fitur Utama:**
        - Upload gambar JPG/PNG
        - Deteksi wajah otomatis
        - Download hasil deteksi
        """)
    else:
        col1, col2 = st.columns([1,1])
        with col1: st.image("foto_saya.jpg", width=200)
        with col2:
            st.info("""
            **Nama:** Heru Bagus Cahyo  
            **Jurusan:** Statistika  
            **Email:** herubagusapk@gmail.com  
            """)

# ======================================
# Halaman Deteksi Wajah
# ======================================
def show_detect(model):
    go_home_button()  # Tombol kembali
    st.markdown("<h1 style='text-align: center; color:#00e0ff;'>😃 Face Detection</h1>", unsafe_allow_html=True)

    pilih_input = st.radio("Pilih Input:", ["🖼️ Upload Gambar", "📷 Kamera"], horizontal=True)

    if pilih_input == "🖼️ Upload Gambar":
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpg","jpeg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("Mendeteksi..."):
                results = model(image, conf=0.25)
                result_image = Image.fromarray(results[0].plot()[..., ::-1])
            col1, col2 = st.columns(2)
            with col1: st.image(image, caption="Gambar Asli", use_container_width=True)
            with col2: st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
            st.success(f"Wajah terdeteksi: {len(results[0].boxes)}")
            st.download_button("💾 Download", data=get_downloadable_image(result_image), file_name="hasil.png", mime="image/png")

    else:  # Mode Kamera
        col1, col2 = st.columns(2)
        with col1: camera_input = st.camera_input("Ambil Foto")
        with col2:
            if camera_input:
                image = Image.open(camera_input).convert("RGB")
                image = PIL.ImageOps.exif_transpose(image)
                with st.spinner("Mendeteksi..."):
                    results = model(image, conf=0.25)
                    result_image = Image.fromarray(results[0].plot()[..., ::-1])
                st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
                st.success(f"Wajah terdeteksi: {len(results[0].boxes)}")
                st.download_button("💾 Download", data=get_downloadable_image(result_image), file_name="hasil.png", mime="image/png")

# ======================================
# Routing Halaman
# ======================================
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "about":
    show_about()
elif st.session_state.page == "detect":
    show_detect(model)

# ======================================
# Footer
# ======================================
st.markdown("""
<footer>
🤖 YOLO Face Detection | Heru Bagus Cahyo
<hr style="border:0.5px solid #00e0ff; margin:10px 0;">
</footer>
""", unsafe_allow_html=True)
