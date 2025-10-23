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
# Fungsi tambahan
# ======================================
def get_downloadable_image(np_img):
    image = Image.fromarray(np_img)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# Letterbox resize agar aspect ratio tetap
def letterbox_image(img, target_size=(640,640)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # grey padding
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
# Sidebar Navigation
# ======================================
page = st.sidebar.radio("Menu", ["Deteksi Wajah", "About Me"])

# ======================================
# About Me
# ======================================
if page == "About Me":
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
# Deteksi Wajah
# ======================================
elif page == "Deteksi Wajah":
    st.markdown("<h1>YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='neon-name'>👨‍💻 Heru Bagus Cahyo</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtext'>Detect faces instantly with YOLO AI — Fast, Accurate, and Powerful.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    detect_button = st.button("🚀 Detect Faces")

    if detect_button and uploaded_file:
        # Cek ukuran file
        if uploaded_file.size > 20*1024*1024:  # 20 MB
            st.warning("⚠️ File terlalu besar, maksimal 20 MB")
        else:
            # Buka gambar dan konversi ke RGB
            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)
            # Optional: histogram equalization untuk wajah gelap
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.equalizeHist(img_gray)
            img_np = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            # Letterbox resize
            img_np = letterbox_image(img_np, target_size=(640,640))

            # Detect
            with st.spinner("Detecting faces... 🔍"):
                start_time = time.time()
                results = model(img_np, conf=0.15, iou=0.3)  # conf lebih rendah untuk wajah jauh/dekat
                inference_time = time.time() - start_time

            result_img = results[0].plot()
            boxes = results[0].boxes.xyxy

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.image(result_img, caption="Detection Result", use_container_width=True)
            st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)

            # Download button
            st.download_button(
                label="💾 Download Detection Result",
                data=get_downloadable_image(result_img),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

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
            st.markdown("</div>", unsafe_allow_html=True)
