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
}
.stImage > img { 
    border-radius: 12px; 
    transition: all 0.3s ease;
}
.stImage > img:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px #00e0ff80;
}
.info-box {
    background: linear-gradient(90deg, #151a28, #1e2440);
    border-radius: 12px;
    padding: 14px 20px;
    color: #bcd4ff;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Fungsi Utility
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
# Load Model YOLO
# ======================================
@st.cache_resource
def load_yolo_model():
    model_path = "model/Cahyo_Laporan4.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model {model_path} tidak ditemukan!")
        return None

model = load_yolo_model()

# ======================================
# Session State untuk Navigasi
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "home"  # default halaman awal

# ======================================
# Header
# ======================================
st.markdown("<h1>YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='neon-name'>👨‍💻 Heru Bagus Cahyo</div>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Detect faces instantly with YOLO AI — Fast, Accurate, and Futuristic.</p>", unsafe_allow_html=True)

# ======================================
# Navigasi Tombol
# ======================================
col1, col2 = st.columns([1,1])
with col1:
    if st.button("About Me"):
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
        # Inisialisasi session_state jika belum ada
        if "about_option" not in st.session_state:
            st.session_state.about_option = "Tentang Website"  # default halaman website
        # Radio button untuk sub-about
        about_selection = st.radio(
            "Pilih:", 
            ["Tentang Website", "Tentang Penulis"], 
            index=0 if st.session_state.about_option == "Tentang Website" else 1
        )
        st.session_state.about_option = about_selection  # update state

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
    st.markdown("<h2>YOLO Face Detection</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    detect_button = st.button("🚀 Detect Faces")

    if detect_button and uploaded_file:
        if uploaded_file.size > 20*1024*1024:
            st.warning("⚠️ File terlalu besar, maksimal 20 MB")
        else:
            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)

            # ======================================
            # Preprocessing Gambar
            # ======================================
            # Histogram equalization untuk wajah gelap
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.equalizeHist(img_gray)
            img_np_eq = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # Letterbox resize agar tetap aspect ratio
            img_np_resized = letterbox_image(img_np_eq, target_size=(640,640))

            # ======================================
            # Deteksi YOLO
            # ======================================
            with st.spinner("Detecting faces... 🔍"):
                start_time = time.time()
                results = model(img_np_resized, conf=0.15, iou=0.3)
                inference_time = time.time() - start_time

            result_img = results[0].plot()
            boxes = results[0].boxes.xyxy

            # ======================================
            # Tampilkan hasil Before / After
            # ======================================
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            col_before, col_after = st.columns(2)
            with col_before:
                st.image(img, caption="Before Detection", use_column_width=True)
            with col_after:
                st.image(result_img, caption="After Detection", use_column_width=True)
            
            st.markdown(f"<div class='info-box'>🕒 Inference Time: {inference_time:.2f} seconds</div>", unsafe_allow_html=True)

            # ======================================
            # Tombol Download
            # ======================================
            st.download_button(
                label="💾 Download Detection Result",
                data=get_downloadable_image(result_img),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

            # ======================================
            # Tampilkan Crop Wajah Detected
            # ======================================
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
# ======================================
# Footer Futuristik
# ======================================
st.markdown("""
<style>
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
.stButton>button {
    transition: all 0.3s ease-in-out;
}
.stButton>button:active {
    transform: scale(0.97);
    box-shadow: 0 0 25px #7a00ff80 inset;
}
.result-card:hover {
    box-shadow: 0 0 25px #00e0ff60;
    transform: translateY(-3px);
}
@media only screen and (max-width: 768px) {
    h1 { font-size: 2rem; }
    .stButton>button { width: 100%; margin-bottom: 10px; }
    .stImage img { width: 100% !important; }
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
</style>
""", unsafe_allow_html=True)

# ======================================
# Neon Animation untuk Judul Deteksi
# ======================================
st.markdown("""
<style>
@keyframes glow {
    0% { text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff; }
    50% { text-shadow: 0 0 15px #00e0ff, 0 0 25px #7a00ff; }
    100% { text-shadow: 0 0 5px #00e0ff, 0 0 10px #7a00ff; }
}
.neon-title {
    animation: glow 1.8s infinite alternate;
    font-weight: 800;
    font-size: 2.2rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Halaman About
# ======================================
if st.session_state.page == "about":
    st.markdown('<h1 class="neon-title">About This App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtext">Learn more about this web application and its creator.</p>', unsafe_allow_html=True)
    
    # Navigasi vertikal sub-about
    col_nav, col_content = st.columns([1,3])
    with col_nav:
        # index=0 -> About Website, index=1 -> About Author
        about_option = st.radio("Pilih:", ["Tentang Website", "Tentang Penulis"], index=0) 
    
    with col_content:
        if about_option == "Tentang Website":
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
# Animasi Glow pada Cards
# ======================================
st.markdown("""
<style>
.result-card {
    transition: all 0.4s ease-in-out;
}
.result-card:hover {
    box-shadow: 0 0 25px #00e0ff80, 0 0 50px #7a00ff60;
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Notifikasi Floating
# ======================================
if 'notification_shown' not in st.session_state:
    st.session_state.notification_shown = False

if not st.session_state.notification_shown:
    st.info("⚡ Tip: Klik 'Deteksi Wajah' untuk mulai mendeteksi wajah secara real-time!")
    st.session_state.notification_shown = True

# ======================================
# Responsif untuk tampilan mobile / tablet
# ======================================
st.markdown("""
<style>
@media only screen and (max-width: 1024px) {
    .stColumns { flex-direction: column !important; }
    .stButton>button { margin-bottom: 15px; width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Footer
# ======================================
st.markdown("""
<footer>
    🤖 YOLO Face Detection Dashboard | Created by <b>Heru Bagus Cahyo</b><br>
    Powered by <b>Streamlit</b> & <b>Ultralytics YOLOv8</b> | UI/UX Futuristic Neon Glow
</footer>
""", unsafe_allow_html=True)
