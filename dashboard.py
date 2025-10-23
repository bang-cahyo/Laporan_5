# ======================================
# app.py - Bagian 1
# ======================================
import streamlit as st
from ultralytics import YOLO
import os
from pages import page_about, page_detect  # memanggil modul halaman
from utils import get_downloadable_image, letterbox_image

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
body, .stApp {
    background: radial-gradient(circle at top left, #0b0f19, #0d1532, #121b3e);
    color: #eaeaea;
    font-family: 'Poppins', sans-serif;
    transition: all 0.3s ease;
}
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}
h1 {
    font-size: 3rem;
    background: linear-gradient(90deg, #00e0ff, #7a00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #00e0ff, #7a00ff);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 0.7em 2em;
    margin: 5px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    box-shadow: 0 0 25px #00e0ff80;
    transform: translateY(-2px);
}
.result-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    margin-top: 20px;
}
.info-box {
    background: linear-gradient(90deg, #151a28, #1e2440);
    border-radius: 12px;
    padding: 14px 20px;
    color: #bcd4ff;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

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
# Session State untuk Navigasi
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ======================================
# Header
# ======================================
st.markdown("<h1>YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='neon-name'>👨‍💻 Heru Bagus Cahyo</div>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Detect faces instantly with YOLO AI — Fast, Accurate, and Futuristic.</p>", unsafe_allow_html=True)

# ======================================
# Tombol Navigasi Halaman
# ======================================
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("About Me", key="btn_about"):
        st.session_state.page = "about"
with col2:
    if st.button("Deteksi Wajah", key="btn_detect"):
        st.session_state.page = "detect"
# ======================================
# page_about.py - Halaman About
# ======================================
import streamlit as st
from PIL import Image

def show_about():
    st.markdown('<h1 class="neon-title">About This App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Learn more about this web application and its creator.</p>', unsafe_allow_html=True)

    # Navigasi vertikal untuk sub-about
    col_nav, col_content = st.columns([1, 3])

    with col_nav:
        if "about_option" not in st.session_state:
            st.session_state.about_option = "Tentang Website"  # default

        about_option = st.radio(
            "Pilih:",
            ["Tentang Website", "Tentang Penulis"],
            index=0 if st.session_state.about_option == "Tentang Website" else 1,
            key="about_radio"
        )
        st.session_state.about_option = about_option

    with col_content:
        if st.session_state.about_option == "Tentang Website":
            st.markdown("""
            **Tentang Website YOLO Face Detection**

            Website ini dibuat untuk mendeteksi wajah pada gambar menggunakan model **YOLOv8** yang sudah dilatih khusus untuk wajah manusia.
            Tujuan website ini adalah memudahkan pengguna mendeteksi wajah secara cepat dan akurat, tanpa perlu menginstal software tambahan atau memahami pemrograman.

            **Fitur Utama:**
            - Upload gambar format JPG, JPEG, atau PNG
            - Deteksi wajah otomatis, menampilkan hasil Before/After secara berdampingan
            - Download hasil deteksi wajah dalam format PNG
            - Tampilan UI futuristik dengan animasi neon untuk pengalaman pengguna yang menarik
            """)
        else:
            col1_bio, col2_bio = st.columns([1,1])
            with col1_bio:
                st.image("foto_saya.jpg", caption="Heru Bagus Cahyo", width=200)
            with col2_bio:
                st.info("""
                **Nama:** Heru Bagus Cahyo  
                **Jurusan:** Statistika  
                **Angkatan:** 2022  
                **Email:** herubagusapk@gmail.com  
                **Instagram:** @herubaguscahyo
                """)
# ======================================
# page_detect.py - Halaman Deteksi Wajah
# ======================================
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
from utils import letterbox_image, get_downloadable_image

def show_detect(model):
    st.markdown('<h2 class="neon-title">YOLO Face Detection</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    detect_button = st.button("🚀 Detect Faces")

    if detect_button and uploaded_file:
        if uploaded_file.size > 20*1024*1024:
            st.warning("⚠️ File terlalu besar, maksimal 20 MB")
            return

        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # ======================================
        # Preprocessing Gambar
        # ======================================
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        img_np_eq = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_np_resized = letterbox_image(img_np_eq, target_size=(640,640))

        # ======================================
        # Deteksi YOLO
        # ======================================
        with st.spinner("Detecting faces... 🔍"):
            start_time = time.time()
            results = model(img_np_resized, conf=0.15, iou=0.3)
            inference_time = time.time() - start_time

        result_img = results[0].plot()
        boxes = results[0].boxes.xyxy

        # ======================================
        # Tampilkan Before / After
        # ======================================
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col_before, col_after = st.columns(2)
        with col_before:
            st.image(img, caption="Before Detection", use_column_width=True)
        with col_after:
            st.image(result_img, caption="After Detection", use_column_width=True)

        st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)

        # ======================================
        # Tombol Download
        # ======================================
        st.download_button(
            label="💾 Download Detection Result",
            data=get_downloadable_image(result_img),
            file_name="hasil_deteksi_wajah.png",
            mime="image/png"
        )

        # ======================================
        # Tampilkan Crop Wajah Detected
        # ======================================
        if len(boxes) > 0:
            st.markdown("### Detected Faces")
            face_cols = st.columns(min(4, len(boxes)))
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = img_np[y1:y2, x1:x2]
                face_img = Image.fromarray(face_crop)
                face_cols[i % len(face_cols)].image(face_img, caption=f"Face {i+1}", width=160)
        else:
            st.warning("⚠️ No faces detected in this image.")

    # ======================================
    # Footer Futuristik
    # ======================================
    st.markdown("""
    <footer>
        🤖 YOLO Face Detection Dashboard | Created by <b>Heru Bagus Cahyo</b><br>
        Powered by <b>Streamlit</b> & <b>Ultralytics YOLOv8</b> | UI/UX Futuristic Neon Glow
    </footer>
    """, unsafe_allow_html=True)
