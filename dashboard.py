import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import io

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
# Sidebar - Toggle Mode
# ======================================
st.sidebar.title("⚙️ Pengaturan")
mode = st.sidebar.radio("🌗 Mode Tampilan", ["Dark Mode", "Light Mode"])
uploaded_file = st.sidebar.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.info("Unggah gambar, sistem akan mendeteksi wajah secara otomatis.")

# ======================================
# CSS Dasar + Tema Dinamis
# ======================================
def apply_theme(mode: str):
    if mode == "Dark Mode":
        bg = "#0d1117"
        text = "#f0f0f0"
        card = "#161b22"
        accent = "#00c8ff"
    else:
        bg = "#f9f9f9"
        text = "#1c1c1c"
        card = "#ffffff"
        accent = "#0078ff"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        .stApp {{
            background-color: {bg};
            color: {text};
            font-family: 'Poppins', sans-serif;
            transition: all 0.5s ease-in-out;
        }}
        h1, h2, h3, h4, h5 {{
            color: {accent};
        }}
        .face-container {{
            padding: 10px;
            border-radius: 10px;
            background-color: {card};
            box-shadow: 0 0 8px rgba(0,0,0,0.3);
            margin-bottom: 15px;
        }}
        .shimmer {{
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
        }}
        @keyframes shimmer {{
            0% {{background-position: 200% 0;}}
            100% {{background-position: -200% 0;}}
        }}
        </style>
    """, unsafe_allow_html=True)

apply_theme(mode)

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
        st.markdown("### 🖼️ Gambar Asli")
        st.image(img, use_container_width=True, caption="Gambar input dari pengguna")

    with col2:
        st.markdown("### 📦 Hasil Deteksi Wajah")

        shimmer_placeholder = st.empty()
        shimmer_placeholder.markdown("<div class='shimmer'></div>", unsafe_allow_html=True)

        # Deteksi YOLO
        start_time = time.time()
        results = model(img_cv2)
        inference_time = time.time() - start_time

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
            st.markdown("#### 🔍 Wajah Terdeteksi:")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = img_cv2[y1:y2, x1:x2]
                face_pil = Image.fromarray(face_crop)

                with st.container():
                    st.markdown(f"<div class='face-container'>Wajah {i+1}</div>", unsafe_allow_html=True)
                    st.image(face_pil, width=180, caption=f"Wajah {i+1}")
else:
    st.markdown("<p>⬆️ Unggah gambar di panel kiri untuk memulai deteksi wajah.</p>", unsafe_allow_html=True)

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & YOLO — by Cahyo</p>",
    unsafe_allow_html=True
)
