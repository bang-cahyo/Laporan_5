import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import time
import os
import cv2

# ======================================
# Konfigurasi Halaman
# ======================================
st.set_page_config(
    page_title="YOLO Face Detection by Heru Bagus Cahyo",
    page_icon="🤖",
    layout="wide"
)

# ======================================
# CSS
# ======================================
st.markdown("""
<style>
body, .stApp {
    background: radial-gradient(circle at top left, #0b0f19, #0d1532, #121b3e);
    color: #eaeaea;
    font-family: 'Poppins', sans-serif;
}
h1 {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #00e0ff, #7a00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background: linear-gradient(90deg, #00e0ff, #7a00ff);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 0.6em 2em;
    margin-right: 10px;
}
.stButton>button:hover {
    box-shadow: 0 0 20px #00e0ff80;
    transform: translateY(-2px);
}
.result-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
}
.stImage > img { border-radius: 12px; }
.info-box {
    background: linear-gradient(90deg, #151a28, #1e2440);
    border-radius: 10px;
    padding: 12px 18px;
    color: #bcd4ff;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Fungsi
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
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top:top+nh, left:left+nw, :] = img_resized
    return canvas

# ======================================
# Load Model YOLO
# ======================================
@st.cache_resource
def load_yolo_model():
    if os.path.exists("model/Cahyo_Laporan4.pt"):
        return YOLO("model/Cahyo_Laporan4.pt")
    else:
        st.error("Model Cahyo_Laporan4.pt tidak ditemukan!")
        return None

model = load_yolo_model()

# ======================================
# Tombol Navigasi
# ======================================
st.markdown("<h1>Dashboard Navigation</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1,1])
with col1:
    show_about = st.button("About Me")
with col2:
    show_detect = st.button("Deteksi Wajah")

# ======================================
# About Me & About Website
# ======================================
if "page" not in st.session_state:
    st.session_state.page = None

col1, col2 = st.columns([1,1])
with col1:
    if st.button("About Me"):
        st.session_state.page = "about"
with col2:
    if st.button("Deteksi Wajah"):
        st.session_state.page = "detect"

# Pilih halaman berdasarkan session_state
if st.session_state.page == "about":
    st.markdown("<h1>About</h1>", unsafe_allow_html=True)

    # Sub-menu About
    col_nav, col_content = st.columns([1,3])
    with col_nav:
        about_option = st.radio("Pilih:", ["Tentang Penulis", "Tentang Website"])

    with col_content:
        if about_option == "Tentang Penulis":
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
        else:
            st.markdown("""
            **Tentang Website YOLO Face Detection**  

            Website ini dibuat untuk mendeteksi wajah pada gambar menggunakan model **YOLOv8**.  
            Tujuan: memudahkan pengguna mendeteksi wajah cepat dan akurat.  
            """)

# ======================================
# Deteksi Wajah
# ======================================
elif show_detect:
    st.markdown("<h1>YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    detect_button = st.button("🚀 Detect Faces")

    if detect_button and uploaded_file:
        if uploaded_file.size > 20*1024*1024:
            st.warning("⚠️ File terlalu besar, maksimal 20 MB")
        else:
            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)

            # Histogram equalization
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.equalizeHist(img_gray)
            img_np_eq = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            img_np_resized = letterbox_image(img_np_eq, target_size=(640,640))

            with st.spinner("Detecting faces... 🔍"):
                start_time = time.time()
                results = model(img_np_resized, conf=0.15, iou=0.3)
                inference_time = time.time() - start_time

            result_img = results[0].plot()
            boxes = results[0].boxes.xyxy

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            col_before, col_after = st.columns(2)
            with col_before:
                st.image(img, caption="Before Detection", use_column_width=True)
            with col_after:
                st.image(result_img, caption="After Detection", use_column_width=True)
            st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)

            st.download_button(
                label="💾 Download Detection Result",
                data=get_downloadable_image(result_img),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

# ======================================
# Footer
# ======================================
st.markdown("""
<footer>
    🤖 YOLO Face Detection Dashboard | Created by <b>Heru Bagus Cahyo</b><br>
    Powered by Streamlit & Ultralytics YOLOv8
</footer>
""", unsafe_allow_html=True)
