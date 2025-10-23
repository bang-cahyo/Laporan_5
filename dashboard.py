import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import time

# ======================================
# Konfigurasi Halaman
# ======================================
st.set_page_config(
    page_title="YOLO Face Detection by Heru Bagus Cahyo",
    page_icon="🤖",
    layout="wide"
)

# ======================================
# CSS — Tampilan Futuristik + Neon Typing + Responsif
# ======================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body, .stApp {
    background: radial-gradient(circle at top left, #0b0f19, #0d1532, #121b3e);
    color: #eaeaea;
    font-family: 'Poppins', sans-serif;
}

/* Hero Title */
h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e0ff, #7a00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtext */
.subtext {
    font-size: 1.1rem;
    color: #b0b0b0;
    margin-bottom: 25px;
}

/* Neon Typing Animation */
@keyframes typing {
  from { width: 0 }
  to { width: 100% }
}
@keyframes blink {
  50% { border-color: transparent }
}
.neon-name {
  font-size: 1.2rem;
  color: #00e0ff;
  font-weight: 700;
  border-right: 2px solid #00e0ff;
  white-space: nowrap;
  overflow: hidden;
  width: 0;
  animation: typing 2s steps(22) forwards, blink 0.75s step-end infinite;
  margin-bottom: 15px;
}

/* Credit name (static) */
.credit {
    font-size: 1rem;
    font-weight: 600;
    color: #00e0ff;
    text-shadow: 0 0 8px rgba(0, 224, 255, 0.6);
    margin-bottom: 15px;
}

/* Tombol */
.stButton>button {
    background: linear-gradient(90deg, #00e0ff, #7a00ff);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 0.8em 2em;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    box-shadow: 0 0 20px #00e0ff80;
    transform: translateY(-2px);
}

/* Card hasil deteksi */
.result-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
}

/* Gambar hasil */
.stImage > img {
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}

/* Info box */
.info-box {
    background: linear-gradient(90deg, #151a28, #1e2440);
    border-radius: 10px;
    padding: 12px 18px;
    color: #bcd4ff;
    font-size: 0.95rem;
    margin-top: 10px;
    text-align: center;
}

/* Footer */
footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 0.9rem;
}

/* Responsif untuk HP */
@media (max-width: 768px){
    h1 { font-size: 2rem; }
    .subtext { font-size: 1rem; }
    .stButton>button { padding: 0.6em 1.5em; font-size: 0.9rem; }
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Fungsi tambahan untuk download hasil deteksi
# ======================================
def get_downloadable_image(np_img):
    image = Image.fromarray(np_img)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# ======================================
# Load Model YOLO
# ======================================
@st.cache_resource
def load_yolo_model():
    return YOLO("model/Cahyo_Laporan4.pt")

model = load_yolo_model()

# ======================================
# Bagian UI — Tampilan Utama
# ======================================
# ======================================
# Sidebar Navigation
# ======================================
page = st.sidebar.radio("Menu", ["Deteksi Wajah", "About Me"])

if page == "About Me":
    # Layout 2 kolom untuk foto dan biodata
    col1_bio, col2_bio = st.columns([1,1])
    with col1_bio:
        # Gunakan URL foto agar aman di Streamlit Cloud
        st.image("https://i.ibb.co/2c8v4P7/foto-saya.jpg", caption="Heru Bagus Cahyo", width=200)
    with col2_bio:
        st.info("""
        **Nama:** Heru Bagus Cahyo  
        **Jurusan:** Statistika  
        **Angkatan:** 2022  
        **Email:** herubagusapk@gmail.com  
        **Instagram:** @herubaguscahyo  
        """)

elif page == "Deteksi Wajah":
    st.markdown("<h1>YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='neon-name'>👨‍💻 Heru Bagus Cahyo</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtext'>Detect faces instantly with YOLO AI — Fast, Accurate, and Powerful.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    detect_button = st.button("🚀 Detect Faces")

    if detect_button and uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        with st.spinner("Detecting faces... 🔍"):
            start_time = time.time()
            results = model(img_np)
            inference_time = time.time() - start_time

        result_img = results[0].plot()
        boxes = results[0].boxes.xyxy

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.image(result_img, caption="Detection Result", use_container_width=True)
        st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)

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
    elif not uploaded_file:
        st.info("📁 Please upload an image to begin detection.")

with col2:
    st.empty()

# ======================================
# Deteksi Wajah
# ======================================
if detect_button and uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    with st.spinner("Detecting faces... 🔍"):
        start_time = time.time()
        results = model(img_np)
        inference_time = time.time() - start_time

    result_img = results[0].plot()
    boxes = results[0].boxes.xyxy

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.image(result_img, caption="Detection Result", use_container_width=True)
    st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)

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

elif not uploaded_file:
    st.info("📁 Please upload an image to begin detection.")

# ======================================
# Footer — Credit Pembuat
# ======================================
st.markdown("""
<footer>
    🤖 YOLO Face Detection Dashboard | Created with  by <b>Heru Bagus Cahyo</b> <br>
    Powered by Streamlit & Ultralytics YOLOv8
</footer>
""", unsafe_allow_html=True)
