import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import time
import cv2

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Deteksi Wajah Otomatis",
    page_icon="🧠",
    layout="wide",
)

# ======================================================
# GAYA DARK MODE ELEGAN
# ======================================================
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #f8f9fa;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        color: #00ccff;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0077ff, #00ccff);
        color: white;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00ccff, #0077ff);
        transform: scale(1.03);
    }
    .uploadedFile {
        border-radius: 10px;
        background-color: #1e222b;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL YOLO
# ======================================================
@st.cache_resource
def load_model():
    model = YOLO("model/Cahyo_Laporan4.pt")  # ganti dengan modelmu
    return model

yolo_model = load_model()

# ======================================================
# FUNGSI UTILITAS
# ======================================================
def get_downloadable_image(image_array):
    """Konversi hasil deteksi menjadi file PNG yang bisa diunduh."""
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================================================
# ANTARMUKA UTAMA
# ======================================================
st.markdown("<h1 style='text-align:center;'>🧠 Dashboard Deteksi Wajah Otomatis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Deteksi wajah secara otomatis menggunakan model YOLOv8 dan Streamlit</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📂 Unggah gambar wajah (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(img)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🖼️ Gambar Asli")
        st.image(img, use_container_width=True, caption="Gambar yang diunggah")

    with col2:
        st.markdown("### 📦 Hasil Deteksi Wajah")

        progress = st.progress(0)
        status_text = st.empty()

        start_time = time.time()
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        status_text.text("🔍 Mendeteksi wajah...")

        results = yolo_model(img_cv2)
        inference_time = time.time() - start_time
        progress.empty()
        status_text.text("✅ Deteksi selesai!")

        # Tampilkan hasil deteksi
        result_img = results[0].plot()
        st.image(result_img, use_container_width=True)
        st.success(f"🕒 Waktu inferensi: {inference_time:.2f} detik")

        # Tombol unduh hasil deteksi
        st.download_button(
            label="💾 Unduh Hasil Deteksi",
            data=get_downloadable_image(result_img),
            file_name="hasil_deteksi_wajah.png",
            mime="image/png"
        )

    # ======================================================
    # TAMPILKAN WAJAH TERDETEKSI (CROP)
    # ======================================================
    boxes = results[0].boxes.xyxy
    if len(boxes) > 0:
        st.markdown("### 👤 Wajah yang Terdeteksi:")
        cols = st.columns(3)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img_cv2[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_pil = Image.fromarray(face_crop)
                with cols[i % 3]:
                    st.image(face_pil, caption=f"Wajah {i+1}", width=180)
    else:
        st.warning("⚠️ Tidak ada wajah yang terdeteksi pada gambar ini.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ by Cahyo using Streamlit & YOLOv8</p>",
    unsafe_allow_html=True
)
