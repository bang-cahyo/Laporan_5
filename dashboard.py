import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import io
import cv2

# ======================================
# Konfigurasi Tampilan
# ======================================
st.set_page_config(
    page_title="Deteksi Wajah YOLO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# Custom CSS
# ======================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    .stApp {
        transition: all 0.5s ease-in-out;
        font-family: 'Poppins', sans-serif;
    }

    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(10px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    /* Loading shimmer */
    .shimmer {
        height: 250px;
        background: linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.05) 25%,
            rgba(255, 255, 255, 0.15) 50%,
            rgba(255, 255, 255, 0.05) 75%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 10px;
        margin-top: 10px;
    }

    @keyframes shimmer {
        0% {background-position: 200% 0;}
        100% {background-position: -200% 0;}
    }

    .face-container {
        padding: 10px;
        border-radius: 10px;
        background-color: var(--card-bg);
        box-shadow: 0 0 8px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }

    /* Mode warna */
    .light {
        --bg: #f9f9f9;
        --text: #1c1c1c;
        --card-bg: #ffffff;
        --accent: #0078ff;
    }

    .dark {
        --bg: #0d1117;
        --text: #f0f0f0;
        --card-bg: #161b22;
        --accent: #00c8ff;
    }

    body, .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    h1, h2, h3, h4 {
        color: var(--accent);
    }
    </style>
""", unsafe_allow_html=True)

# ======================================
# Toggle Mode
# ======================================
mode = st.sidebar.radio("🌗 Mode Tampilan", ["Dark Mode", "Light Mode"])

theme_class = "dark" if mode == "Dark Mode" else "light"
st.markdown(f"<body class='{theme_class}'>", unsafe_allow_html=True)

# ======================================
# Load Model YOLO
# ======================================
@st.cache_resource
def load_yolo_model():
    return YOLO("model/Cahyo_Laporan4.pt")

model = load_yolo_model()

# ======================================
# Header
# ======================================
st.markdown("<h1 class='fade-in'>🧠 Deteksi Wajah Otomatis</h1>", unsafe_allow_html=True)
st.caption("Gunakan model **YOLO** untuk mendeteksi wajah secara cepat dan akurat.")

# ======================================
# Sidebar
# ======================================
st.sidebar.header("⚙️ Pengaturan")
st.sidebar.info("Unggah gambar, sistem akan mendeteksi wajah secara otomatis.")
uploaded_file = st.sidebar.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ======================================
# Fungsi Download
# ======================================
def get_downloadable_image(image_array):
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================================
# Konten Utama
# ======================================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(img)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<h3 class='fade-in'>🖼️ Gambar Asli</h3>", unsafe_allow_html=True)
        st.image(img, use_container_width=True, caption="Gambar input dari pengguna")

    with col2:
        st.markdown("<h3 class='fade-in'>📦 Hasil Deteksi Wajah</h3>", unsafe_allow_html=True)

        # Efek shimmer sementara
        shimmer_placeholder = st.empty()
        shimmer_placeholder.markdown("<div class='shimmer'></div>", unsafe_allow_html=True)

        # Deteksi YOLO
        start_time = time.time()
        results = model(img_cv2)
        inference_time = time.time() - start_time

        # Tampilkan hasil deteksi
        result_img = results[0].plot()
        shimmer_placeholder.empty()
        st.image(result_img, use_container_width=True, caption="Hasil deteksi wajah")

        st.markdown(f"🕒 **Waktu inferensi:** {inference_time:.2f} detik")

        # Tombol download
        st.download_button(
            label="💾 Unduh Hasil Deteksi",
            data=get_downloadable_image(result_img),
            file_name="hasil_deteksi_wajah.png",
            mime="image/png"
        )

        # Tampilkan crop wajah
        boxes = results[0].boxes.xyxy
        if len(boxes) > 0:
            st.markdown("<h4>🔍 Wajah Terdeteksi:</h4>", unsafe_allow_html=True)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = img_cv2[y1:y2, x1:x2]
                face_pil = Image.fromarray(face_crop)

                st.markdown(f"<div class='face-container fade-in'>Wajah {i+1}</div>", unsafe_allow_html=True)
                st.image(face_pil, width=180, caption=f"Wajah {i+1}")
else:
    st.markdown("<p class='fade-in'>⬆️ Unggah gambar di panel kiri untuk memulai deteksi wajah.</p>", unsafe_allow_html=True)

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & YOLO — by Cahyo</p>",
    unsafe_allow_html=True
)
