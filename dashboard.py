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
.subtext {
    font-size: 1.1rem;
    color: #b0b0b0;
    margin-bottom: 25px;
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
# Session State Default
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "home"  # halaman awal
if "about_option" not in st.session_state:
    st.session_state.about_option = "Tentang Website"

# ======================================
# Header UI/UX
# ======================================
st.markdown("<h1>YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Detect faces instantly with YOLO AI — Fast, Accurate, and Powerful.</div>", unsafe_allow_html=True)

# ======================================
# Navigasi Tombol
# ======================================
col1, col2 = st.columns([1,1])
with col1:
    if st.button("About Me / Website"):
        st.session_state.page = "about"
with col2:
    if st.button("Deteksi Wajah"):
        st.session_state.page = "detect"

# ======================================
# Halaman About
# ======================================
if st.session_state.page == "about":
    st.markdown("<h1>About</h1>", unsafe_allow_html=True)
    
    col_nav, col_content = st.columns([1,3])
    with col_nav:
        st.session_state.about_option = st.radio(
            "Pilih:", 
            ["Tentang Website", "Tentang Penulis"], 
            index=0 if st.session_state.about_option=="Tentang Website" else 1
        )

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

            **Cara Penggunaan:**
            1. Pilih menu **Deteksi Wajah** di atas.
            2. Klik tombol **Upload an image** dan pilih gambar dari perangkat Anda.
            3. Klik tombol **🚀 Detect Faces** untuk memulai deteksi.
            4. Hasil deteksi akan muncul berdampingan: sebelah kiri **Before** (gambar asli), sebelah kanan **After** (gambar dengan bounding box wajah).
            5. Jika ingin menyimpan hasil, klik tombol **Download Detection Result**.

            Website ini dibuat oleh **Heru Bagus Cahyo** menggunakan **Streamlit** dan **Ultralytics YOLOv8**, sehingga dapat berjalan di browser tanpa instalasi tambahan.
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
# Halaman Deteksi Wajah
# ======================================
elif st.session_state.page == "detect":
    st.markdown("<h1>YOLO Face Detection</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
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


st.markdown("""
<footer>
    🤖 YOLO Face Detection Dashboard | Created by <b>Heru Bagus Cahyo</b><br>
    Powered by Streamlit & Ultralytics YOLOv8
</footer>
""", unsafe_allow_html=True)
