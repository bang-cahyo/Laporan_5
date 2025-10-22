import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import time
import cv2

# ======================================================
# KONFIGURASI DASHBOARD
# ======================================================
st.set_page_config(
    page_title="Deteksi Wajah Otomatis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Mode Elegan
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #00ccff;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #00ccff;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL YOLO (CACHED)
# ======================================================
@st.cache_resource
def load_model():
    model = YOLO("model/Cahyo_Laporan4.pt")  # ganti sesuai nama model kamu
    return model

yolo_model = load_model()

# ======================================================
# FUNGSI UTILITAS
# ======================================================
def get_downloadable_image(image_array):
    """Mengonversi array gambar menjadi file PNG agar bisa diunduh."""
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================================================
# ANTARMUKA UTAMA
# ======================================================
st.title("🧠 Dashboard Deteksi Wajah Otomatis")
st.caption("Aplikasi berbasis Streamlit dan YOLO untuk mendeteksi wajah pada gambar")

# Sidebar Menu
st.sidebar.header("📂 Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["Deteksi Wajah", "Penjelasan Fitur Dashboard"])

# ======================================================
# MODE 1: DETEKSI WAJAH
# ======================================================
if menu == "Deteksi Wajah":
    st.subheader("🔍 Unggah Gambar untuk Deteksi Wajah")
    uploaded_file = st.file_uploader("Pilih gambar (format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_cv2 = np.array(img)

        # Dua kolom: kiri (gambar asli), kanan (hasil deteksi)
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)

        with col2:
            st.write("📦 **Hasil Deteksi Wajah**")
            start_time = time.time()
            results = yolo_model(img_cv2)
            inference_time = time.time() - start_time

            # Ambil hasil deteksi
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)
            st.success(f"Waktu inferensi: {inference_time:.2f} detik")

            # Tombol unduh
            st.download_button(
                label="💾 Unduh Hasil Deteksi",
                data=get_downloadable_image(result_img),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

        # ======================================================
        # Crop wajah hasil deteksi (jika ada)
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
            st.warning("⚠️ Tidak ada wajah terdeteksi pada gambar ini.")

# ======================================================
# MODE 2: PENJELASAN FITUR DASHBOARD
# ======================================================
elif menu == "Penjelasan Fitur Dashboard":
    st.header("📘 Penjelasan Fitur Dashboard")

    st.markdown("""
    ### **a. Fitur-fitur utama**
    1. **Unggah Gambar** — Menggunakan `st.file_uploader` untuk mengunggah gambar wajah.
    2. **Deteksi Wajah (YOLO)** — Model YOLO mendeteksi wajah dan menampilkan bounding box.
    3. **Waktu Inferensi** — Mengukur waktu proses model dalam mendeteksi wajah.
    4. **Crop Wajah** — Menampilkan potongan (crop) setiap wajah yang terdeteksi.
    5. **Unduh Hasil Deteksi** — Tombol `st.download_button` untuk menyimpan hasil deteksi.
    6. **Tampilan Dark Mode Elegan** — Menggunakan *custom CSS* agar tampilan lebih profesional.

    ---

    ### **b. Komponen Streamlit yang digunakan**
    | Komponen | Fungsi |
    |-----------|--------|
    | `st.file_uploader` | Mengunggah gambar dari perangkat |
    | `st.image` | Menampilkan gambar dan hasil deteksi |
    | `st.columns` | Membagi layout menjadi dua kolom |
    | `st.download_button` | Mengunduh hasil deteksi |
    | `st.cache_resource` | Menyimpan model YOLO di cache agar tidak diload ulang |
    | `st.success`, `st.warning` | Menampilkan notifikasi hasil |
    | `st.sidebar.radio` | Menu navigasi antar halaman |
    | `st.markdown` | Menulis teks dan tabel deskriptif |

    ---

    ### **c. Alur penggunaan dashboard**
    1. Pengguna membuka aplikasi Streamlit.  
    2. Mengunggah gambar wajah pada bagian “Deteksi Wajah”.  
    3. Model YOLO melakukan inferensi dan menampilkan hasil deteksi.  
    4. Sistem menampilkan waktu inferensi dan hasil crop wajah.  
    5. Pengguna dapat mengunduh gambar hasil deteksi.  
    6. Dashboard menampilkan pesan penutup sebagai akhir proses.

    ---

    ### **Kesimpulan**
    Dashboard ini mampu melakukan deteksi wajah secara cepat dan akurat menggunakan model YOLO, 
    dengan antarmuka interaktif yang dibangun di Streamlit.
    """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & YOLO</p>",
    unsafe_allow_html=True
)
