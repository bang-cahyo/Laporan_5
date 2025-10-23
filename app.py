import streamlit as st
from dashboard import show_detect  # pastikan file show_detect ada di folder project kamu

# =========================
# INISIALISASI MODEL YOLO
# =========================
# Kalau kamu sudah punya model, load di sini
# model = load_model("Cahyo_Laporan4.pt")
model = None  # sementara, biar gak error dulu

# =========================
# HALAMAN ABOUT
# =========================
def show_about():
    st.markdown("<h1 style='text-align:center;'>💫 Tentang Aplikasi</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:justify; color:#d7e3ff; font-size:1.1rem;'>
    Aplikasi ini dikembangkan oleh <b>Heru Bagus Cahyo</b> sebagai implementasi AI berbasis <b>YOLOv8</b> 
    untuk mendeteksi wajah dan ekspresi manusia secara real-time.  
    Dibangun dengan <b>Streamlit</b> agar mudah digunakan dan memiliki tampilan futuristik yang elegan.  
    </div>
    """, unsafe_allow_html=True)

# =========================
# HALAMAN HOME
# =========================
def show_home():
    st.markdown("<h1 class='neon-title' style='text-align:center;'>Welcome to YOLO Face Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#bcd4ff;'>Powered by AI. Designed with style. Built by <b>Heru Bagus Cahyo</b>.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Deskripsi Aplikasi
    st.markdown("""
    <div style='text-align: justify; font-size:1.1rem; color:#d7e3ff;'>
    Aplikasi ini dirancang untuk mendeteksi ekspresi wajah secara otomatis menggunakan model berbasis <b>YOLOv8</b>.
    Dibuat dengan tampilan futuristik yang interaktif, aplikasi ini memudahkan pengguna dalam mengenali ekspresi seperti 
    <b>happy</b>, <b>sad</b>, <b>fear</b>, <b>anger</b>, dan lainnya hanya dengan satu klik.  
    </div>
    """, unsafe_allow_html=True)

    # 3 Fitur Utama
    st.markdown("<h2 class='neon-title' style='margin-top:30px;'>✨ Fitur Unggulan</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='result-card' style='text-align:center;'>
            <h3>⚡ Deteksi Cepat</h3>
            <p>Gunakan YOLOv8 untuk mendeteksi ekspresi wajah dengan kecepatan tinggi dan hasil akurat.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='result-card' style='text-align:center;'>
            <h3>🖼️ Upload Gambar / Kamera</h3>
            <p>Pilih metode input sesuai kebutuhan: unggah gambar atau langsung dari kamera perangkatmu.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='result-card' style='text-align:center;'>
            <h3>🌈 UI Futuristik</h3>
            <p>Tampilan modern bertema neon-glow dengan animasi lembut untuk pengalaman pengguna terbaik.</p>
        </div>
        """, unsafe_allow_html=True)

    # Tombol Navigasi Cepat
    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB, colC = st.columns([1, 1, 1])
    with colB:
        detect = st.button("🚀 Mulai Deteksi Wajah", use_container_width=True)
        about = st.button("💫 Tentang Aplikasi", use_container_width=True)

        if detect:
            st.session_state.page = "detect"
            st.rerun()
        elif about:
            st.session_state.page = "about"
            st.rerun()

# =========================
# SIDEBAR NAVIGASI
# =========================
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

    menu = st.radio("Navigasi:", ["🏠 Home", "🧍 About Me", "📷 Deteksi Wajah"], index=0)

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if menu == "🏠 Home":
        st.session_state.page = "home"
    elif menu == "🧍 About Me":
        st.session_state.page = "about"
    elif menu == "📷 Deteksi Wajah":
        st.session_state.page = "detect"

# =========================
# TAMPILKAN HALAMAN SESUAI MENU
# =========================
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "about":
    show_about()
elif st.session_state.page == "detect":
    show_detect(model)
else:
    show_home()
