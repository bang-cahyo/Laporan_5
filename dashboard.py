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
# Sidebar Navigasi Vertikal Elegan & Fungsional
# ======================================
with st.sidebar:
    st.markdown("""
        <style>
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #00e0ff;
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 0 0 10px #00e0ff, 0 0 20px #7a00ff;
        }
        .sidebar-subtext {
            color: #bcd4ff;
            font-size: 0.9rem;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>🤖 YOLO Face Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtext'>by Heru Bagus Cahyo</div>", unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi:",
        ["🏠 Home", "🧍 About", "📷 Deteksi Wajah"],
        index=0
    )

    if menu == "🏠 Home":
        st.session_state.page = "home"
    elif menu == "🧍 About":
        st.session_state.page = "about"
    elif menu == "📷 Deteksi Wajah":
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
from io import BytesIO
from PIL import Image
import streamlit as st

def show_detect(model):
    # ==============================
    # 🔹 Header Futuristik & Deskripsi Model
    # ==============================
    st.markdown("<h1 style='text-align: center; color:#00e0ff;'>😃 Face Expression Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #bcd4ff;'>By Heru Bagus Cahyo</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ==============================
    # 🔹 Pilihan Input (Upload / Kamera)
    # ==============================
    pilih_input = st.radio("Pilih Sumber Input:", ["🖼️ Upload Gambar", "📷 Gunakan Kamera"], horizontal=True)

    # ======================================
    # 🖼️ MODE UPLOAD GAMBAR
    # ======================================
    if pilih_input == "🖼️ Upload Gambar":
        uploaded_file = st.file_uploader("📁 Upload Gambar", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Konversi file ke image PIL
            image = Image.open(uploaded_file)

            # Pastikan format RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Deteksi wajah otomatis
            with st.spinner("🔍 Mendeteksi wajah..."):
                results = model(image, conf=0.25)  # threshold sensitif
                result_image = results[0].plot()  # hasil berupa array BGR
                result_image = Image.fromarray(result_image[..., ::-1])  # ubah ke RGB

            # ======================================
            # 🔹 TAMPILKAN BEFORE - AFTER
            # ======================================
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

            with col2:
                st.image(result_image, caption="✅ Hasil Deteksi", use_container_width=True)

            # ======================================
            # 🔹 INFORMASI DETEKSI
            # ======================================
            num_faces = len(results[0].boxes)
            st.success(f"✅ Jumlah wajah terdeteksi: {num_faces}")

            # Ekstraksi ekspresi dari hasil deteksi
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                detected_expressions = [
                    results[0].names[int(cls)] for cls in results[0].boxes.cls
                ]
                unique_expressions = sorted(set(detected_expressions))
                st.markdown(
                    f"""
                    <div style='background-color:#0f172a; padding:12px; border-radius:10px;
                    border:1px solid #00e0ff; color:#00e0ff; text-align:center;'>
                    😃 <b>Ekspresi Terdeteksi:</b> {', '.join(unique_expressions)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("😕 Tidak ada ekspresi wajah terdeteksi.")

            # ======================================
            # 🔹 DOWNLOAD HASIL DETEKSI
            # ======================================
            def get_downloadable_image(img):
                buf = BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                return byte_im

            st.download_button(
                label="💾 Download Hasil Deteksi",
                data=get_downloadable_image(result_image),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

    # ==============================
    # Gunakan Kamera
    # ==============================
    elif pilih_input == "📷 Gunakan Kamera":
        st.info("📸 Gunakan kamera untuk mengambil gambar dan deteksi ekspresi otomatis.")
        col_cam, col_after = st.columns(2)

        with col_cam:
            cam_image = st.camera_input("Ambil gambar dari kamera")

        if cam_image:
            img = Image.open(cam_image).convert("RGB")
            img_np = np.array(img)
            img_np_resized = letterbox_image(img_np, target_size=(640, 640))

            # Jalankan deteksi otomatis
            with st.spinner("Mendeteksi ekspresi wajah... 🔍"):
                start_time = time.time()
                results = model(img_np_resized, conf=0.25, iou=0.3)
                inference_time = time.time() - start_time

            # Gambar hasil deteksi
            result_img_resized = results[0].plot()
            h, w = img_np.shape[:2]
            result_img_resized = cv2.resize(result_img_resized, (w, h))

            with col_after:
                st.image(result_img_resized, caption="Hasil Deteksi", use_container_width=True)
                st.markdown(f"<div class='info-box'>🕒 Waktu Inference: {inference_time:.2f} detik</div>", unsafe_allow_html=True)

                # Ekstraksi ekspresi
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    detected_expressions = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
                    unique_expressions = sorted(set(detected_expressions))
                    st.markdown(
                        f"<div class='info-box'>😃 Ekspresi Terdeteksi: {', '.join(unique_expressions)}</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("😕 Tidak ada ekspresi wajah terdeteksi.")

                # Tombol download hasil
                st.download_button(
                    label="💾 Download Hasil Deteksi",
                    data=get_downloadable_image(result_img_resized),
                    file_name="hasil_deteksi_kamera.png",
                    mime="image/png"
                )

# ======================================
# Routing Halaman Berdasarkan Sidebar
# ======================================
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
