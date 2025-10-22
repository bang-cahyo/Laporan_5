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
# GAYA DARK MODE MODERN
# ======================================================
st.markdown("""
    <style>
    * {font-family: 'Poppins', sans-serif;}
    body, .stApp {
        background-color: #0d1117;
        color: #f8f9fa;
    }
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 600;
        color: #00c8ff;
        margin-bottom: -10px;
    }
    .sub-title {
        text-align: center;
        color: #aaa;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    .card {
        background-color: #161b22;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(0, 200, 255, 0.05);
        transition: 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 0 35px rgba(0, 200, 255, 0.15);
        transform: scale(1.01);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(90deg, #0077ff, #00c8ff);
        color: white;
        font-weight: 600;
        border: none;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00c8ff, #0077ff);
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL YOLO
# ======================================================
@st.cache_resource
def load_model():
    model = YOLO("model/Cahyo_Laporan4.pt")  # Ganti sesuai lokasi model kamu
    return model

yolo_model = load_model()

# ======================================================
# FUNGSI UTILITAS
# ======================================================
def get_downloadable_image(image_array):
    """Konversi hasil deteksi ke file PNG."""
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1 class='main-title'>🧠 Deteksi Wajah Otomatis</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Deteksi wajah secara cepat dan akurat menggunakan model YOLOv8</p>", unsafe_allow_html=True)
st.markdown("---")

# ======================================================
# UPLOAD GAMBAR
# ======================================================
uploaded_file = st.file_uploader("📂 Unggah gambar wajah (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(img)

    col1, col2 = st.columns([1, 1])

    # ==========================================
    # KIRI: GAMBAR ASLI
    # ==========================================
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Gambar Asli")
        st.image(img, use_container_width=True, caption="Gambar yang diunggah")
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # KANAN: HASIL DETEKSI
    # ==========================================
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📦 Hasil Deteksi Wajah")

        with st.spinner("🔍 Mendeteksi wajah..."):
            start_time = time.time()
            results = yolo_model(img_cv2)
            inference_time = time.time() - start_time

        result_img = results[0].plot()
        st.image(result_img, use_container_width=True)
        st.success(f"✅ Deteksi selesai dalam {inference_time:.2f} detik")

        st.download_button(
            label="💾 Unduh Hasil Deteksi",
            data=get_downloadable_image(result_img),
            file_name="hasil_deteksi_wajah.png",
            mime="image/png"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # MENAMPILKAN WAJAH TERDETEKSI
    # ==========================================
    boxes = results[0].boxes.xyxy
    if len(boxes) > 0:
        st.markdown("<br><div class='card'>", unsafe_allow_html=True)
        st.markdown("### 👤 Wajah yang Terdeteksi")
        cols = st.columns(3)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img_cv2[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_pil = Image.fromarray(face_crop)
                with cols[i % 3]:
                    st.image(face_pil, caption=f"Wajah {i+1}", width=180)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Tidak ada wajah yang terdeteksi pada gambar ini.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#777;'>Made with ❤️ by Cahyo | Powered by Streamlit & YOLOv8</p>",
    unsafe_allow_html=True
)
