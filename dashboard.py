import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import time
import cv2

# ======================================================
# KONFIGURASI DASAR DASHBOARD
# ======================================================
st.set_page_config(
    page_title="Dashboard Deteksi Wajah",
    page_icon="🧠",
    layout="wide",
)

# Custom Dark Mode + Style
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
    </style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL YOLO (dengan cache biar cepat)
# ======================================================
@st.cache_resource
def load_model():
    model = YOLO("model/Cahyo_Laporan4.pt")  # ganti dengan model kamu
    return model

yolo_model = load_model()

# ======================================================
# FUNGSI UTILITAS
# ======================================================
def get_downloadable_image(image_array):
    """Mengonversi array gambar ke file PNG agar bisa diunduh."""
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================================================
# ANTARMUKA UTAMA
# ======================================================
st.title("🧠 Dashboard Deteksi Wajah Otomatis")
st.caption("Aplikasi berbasis YOLO dan Streamlit untuk mendeteksi wajah secara real-time dari gambar yang diunggah.")

# Fitur utama: unggah gambar
uploaded_file = st.file_uploader("📂 Unggah gambar (format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv2 = np.array(img)

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

        # Tombol unduh hasil deteksi
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
# PENJELASAN FITUR DASHBOARD (untuk laporan tugas)
# ======================================================
st.markdown("---")
st.header("📘 Penjelasan Fitur Dashboard")

st.markdown("""
### **a. Fitur-fitur utama**
1. **Unggah Gambar** — Menggunakan `st.file_uploader` untuk memilih gambar dari perangkat pengguna.  
2. **Deteksi Wajah (YOLO)** — Model YOLO mendeteksi area wajah dan menampilkan bounding box di sekitar wajah.  
3. **Waktu Inferensi** — Menampilkan waktu proses deteksi agar pengguna tahu performa model.  
4. **Crop Wajah** — Menampilkan potongan wajah yang berhasil terdeteksi secara otomatis.  
5. **Unduh Hasil Deteksi** — Tombol `st.download_button` untuk menyimpan hasil deteksi ke komputer.  
6. **Tampilan Dark Mode** — Menggunakan *custom CSS* agar dashboard terlihat profesional dan nyaman di mata.

---

### **b. Komponen Streamlit yang digunakan**
| Komponen | Fungsi |
|-----------|--------|
| `st.file_uploader` | Mengunggah gambar dari perangkat |
| `st.image` | Menampilkan gambar dan hasil deteksi |
| `st.columns` | Membagi layout menjadi dua kolom |
| `st.download_button` | Menyediakan tombol untuk mengunduh hasil deteksi |
| `st.cache_resource` | Menyimpan model di cache agar tidak diload ulang |
| `st.success`, `st.warning` | Memberi umpan balik hasil deteksi |
| `st.markdown` | Menulis teks deskriptif dan tabel |

---

### **c. Alur penggunaan dashboard**
1. Pengguna membuka aplikasi Streamlit.  
2. Mengunggah gambar pada bagian “Unggah Gambar”.  
3. Model YOLO melakukan deteksi wajah secara otomatis.  
4. Hasil deteksi dan waktu inferensi ditampilkan di sisi kanan.  
5. Jika wajah terdeteksi, potongan wajah (crop) akan muncul di bawahnya.  
6. Pengguna dapat mengunduh hasil deteksi dalam format PNG.

---

### **Kesimpulan**
Dashboard ini mampu melakukan deteksi wajah secara otomatis menggunakan model YOLO.  
Proses inferensi cepat, hasil visualisasi jelas, dan tampilannya dibuat profesional dengan Streamlit.
""")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & YOLO</p>",
    unsafe_allow_html=True
)
