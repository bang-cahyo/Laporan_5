import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import io
import cv2
from streamlit.components.v1 import html

# ======================================
# Konfigurasi Tampilan Streamlit
# ======================================
st.set_page_config(
    page_title="Deteksi Wajah YOLO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# Custom CSS untuk UI Elegan + Animasi
# ======================================
st.markdown("""
    <style>
    /* Animasi Fade-In */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }

    /* Gaya umum */
    .stApp {
        background-color: #0d1117;
        color: #f0f0f0;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4 {
        color: #00c8ff;
        font-weight: 600;
    }

    .upload-text {
        color: #f0f0f0;
        font-size: 1rem;
    }

    .face-container {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        box-shadow: 0px 0px 10px rgba(0, 200, 255, 0.2);
    }

    .inference {
        font-size: 0.9rem;
        color: #a0a0a0;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================
# Load Model YOLO (Cache)
# ======================================
@st.cache_resource
def load_yolo_model():
    return YOLO("model/Cahyo_Laporan4.pt")

model = load_yolo_model()

# ======================================
# Judul & Deskripsi
# ======================================
st.markdown("<h1 class='fade-in'>🧠 Deteksi Wajah Otomatis</h1>", unsafe_allow_html=True)
st.caption("Aplikasi ini menggunakan model **YOLO** untuk mendeteksi wajah secara otomatis pada gambar yang diunggah.")

# ======================================
# Sidebar
# ======================================
st.sidebar.header("⚙️ Pengaturan")
st.sidebar.info("Unggah gambar wajah di bawah, sistem akan mendeteksi area wajah secara otomatis.")
st.sidebar.markdown("---")

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
# Proses Deteksi
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

        start_time = time.time()
        results = model(img_cv2)
        inference_time = time.time() - start_time

        result_img = results[0].plot()
        st.image(result_img, use_container_width=True, caption="Deteksi wajah oleh YOLO", output_format="PNG")

        st.markdown(f"<p class='inference fade-in'>🕒 Waktu inferensi: <b>{inference_time:.2f} detik</b></p>", unsafe_allow_html=True)

        # Tombol download
        st.download_button(
            label="💾 Unduh Hasil Deteksi",
            data=get_downloadable_image(result_img),
            file_name="hasil_deteksi_wajah.png",
            mime="image/png"
        )

        # Crop wajah terdeteksi
        boxes = results[0].boxes.xyxy
        if len(boxes) > 0:
            st.markdown("<h4 class='fade-in'>🔍 Wajah Terdeteksi:</h4>", unsafe_allow_html=True)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = img_cv2[y1:y2, x1:x2]
                face_pil = Image.fromarray(face_crop)

                with st.container():
                    st.markdown(f"<div class='face-container fade-in'>Wajah {i+1}</div>", unsafe_allow_html=True)
                    st.image(face_pil, width=180, caption=f"Wajah {i+1}")

else:
    st.markdown("<p class='fade-in upload-text'>⬆️ Silakan unggah gambar di panel kiri untuk memulai deteksi wajah.</p>", unsafe_allow_html=True)

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #808080;'>Made with ❤️ using Streamlit & YOLO — by Cahyo</p>",
    unsafe_allow_html=True
)
