# ======================================
# dashboard.py - YOLO Face Detection
# ======================================
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import time
import os
import cv2
from utils import letterbox_image, get_downloadable_image
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
    transition: all 0.4s ease-in-out;
}
.result-card:hover {
    box-shadow: 0 0 25px #00e0ff80, 0 0 50px #7a00ff60;
    transform: translateY(-5px);
}
.info-box {
    background: linear-gradient(90deg, #151a28, #1e2440);
    border-radius: 12px;
    padding: 14px 20px;
    color: #bcd4ff;
    text-align: center;
    margin-top: 10px;
}
.neon-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #00e0ff;
    text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff, 0 0 20px #00e0ff;
}
.subtext {
    font-size: 1rem;
    color: #bcd4ff;
    margin-bottom: 20px;
}
.neon-title {
    animation: glow 1.8s infinite alternate;
    font-weight: 800;
    font-size: 2.2rem;
}
@keyframes glow {
    0% { text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff; }
    50% { text-shadow: 0 0 15px #00e0ff, 0 0 25px #7a00ff; }
    100% { text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff; }
}
footer {
    background: linear-gradient(90deg, #0d0f1a, #1e1f3a);
    padding: 15px 20px;
    border-radius: 12px;
    margin-top: 50px;
    text-align: center;
    font-size: 0.9rem;
    color: #bcd4ff;
    box-shadow: 0 0 20px #00e0ff40;
}
@media only screen and (max-width: 1024px) {
    .stColumns { flex-direction: column !important; }
    .stButton>button { margin-bottom: 15px; width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Utility Functions
# ======================================
def get_downloadable_image(np_img):
    """Convert numpy array to downloadable PNG bytes."""
    image = Image.fromarray(np_img)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def letterbox_image(img, target_size=(640,640)):
    """Resize image keeping aspect ratio with padding."""
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
# Halaman About
# ======================================
def show_about():
    st.markdown('<h1 class="neon-title">About This App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Learn more about this web application and its creator.</p>', unsafe_allow_html=True)

    col_nav, col_content = st.columns([1, 3])
    with col_nav:
        if "about_option" not in st.session_state:
            st.session_state.about_option = "Tentang Website"
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
    
            Website ini dibuat untuk mendeteksi wajah pada gambar secara otomatis menggunakan model **YOLOv8** yang sudah dilatih khusus untuk wajah manusia. 
            Tujuannya adalah memudahkan pengguna mendeteksi wajah tanpa perlu menginstal software tambahan atau memahami pemrograman.
    
            **Fitur Utama:**
            - Upload gambar dalam format JPG, JPEG, atau PNG.
            - Deteksi wajah secara otomatis dengan bounding box.
            - Hasil Before/After ditampilkan berdampingan untuk memudahkan perbandingan.
            - Download hasil deteksi wajah dalam format PNG.
            - UI Futuristik dengan animasi Neon Glow untuk pengalaman interaktif.
    
            **Cara Menggunakan:**
            1. Pilih menu **Deteksi Wajah** di atas.
            2. Upload gambar dari perangkat Anda menggunakan tombol upload.
            3. Klik tombol **🚀 Detect Faces** untuk memulai deteksi.
            4. Hasil deteksi muncul di kolom Before/After, dan dapat diunduh jika diinginkan.
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
def show_detect(model):
    st.markdown('<h2 class="neon-title">YOLO Face Detection</h2>', unsafe_allow_html=True)

    pilih_input = st.radio("Pilih Sumber Input:", ["Upload Gambar", "Gunakan Kamera"])

    if pilih_input == "Upload Gambar":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        detect_button = st.button("🚀 Detect Faces")

        if detect_button and uploaded_file:
            if uploaded_file.size > 20*1024*1024:
                st.warning("⚠️ File terlalu besar, maksimal 20 MB")
                return

            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)

            # Resize YOLO input
            img_np_resized = letterbox_image(img_np, target_size=(640,640))

            with st.spinner("Detecting faces... 🔍"):
                start_time = time.time()
                results = model(img_np_resized, conf=0.15, iou=0.3)
                inference_time = time.time() - start_time

            result_img = results[0].plot()
            # Resize hasil deteksi mengikuti rasio asli input
            result_img_resized = cv2.resize(result_img, (img_np.shape[1], img_np.shape[0]))

            # Tampilkan Before / After sejajar
            col_before, col_after = st.columns(2)
            with col_before:
                st.image(img_np, caption="Before Detection", use_container_width=True)
            with col_after:
                st.image(result_img_resized, caption="After Detection", use_container_width=True)

            st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)
            st.download_button(
                label="💾 Download Detection Result",
                data=get_downloadable_image(result_img_resized),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

    elif pilih_input == "Gunakan Kamera":
        col_result = st.columns(1)[0]  # kolom tunggal untuk hasil

        cam_image = st.camera_input(label="")  # label dikosongkan

        if cam_image:
            img = Image.open(cam_image).convert("RGB")
            img_np = np.array(img)
            h_input, w_input = img_np.shape[:2]

            # Resize untuk YOLO
            img_np_resized = letterbox_image(img_np, target_size=(640,640))

            with st.spinner("Detecting faces... 🔍"):
                start_time = time.time()
                results = model(img_np_resized, conf=0.15, iou=0.3)
                inference_time = time.time() - start_time

            result_img = results[0].plot()
            # Resize hasil deteksi supaya mengikuti rasio kamera
            result_img_resized = cv2.resize(result_img, (w_input, h_input))

            with col_result:
                st.image(result_img_resized, caption="Hasil Deteksi", use_container_width=True)
                st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download Detection Result",
                    data=get_downloadable_image(result_img_resized),
                    file_name="hasil_deteksi_kamera.png",
                    mime="image/png"
                )



# Render halaman sesuai pilihan
if st.session_state.page == "about":
    show_about()
elif st.session_state.page == "detect":
    show_detect(model)
else:
    st.write("Selamat datang di YOLO Face Detection Dashboard!")



# ======================================
# Footer dengan About
# ======================================
st.markdown(f"""
<footer>
    🤖 YOLO Face Detection Dashboard | Created by <b>Heru Bagus Cahyo</b><br>
    Powered by <b>Streamlit</b> & <b>Ultralytics YOLOv8</b> | UI/UX Futuristic Neon Glow
    <hr style="border:0.5px solid #00e0ff; margin:10px 0;">
    <div style="font-size:0.85rem; color:#bcd4ff;">
</footer>
""", unsafe_allow_html=True)
