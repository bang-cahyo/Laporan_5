import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import time

# ======================================
# Konfigurasi Tampilan Streamlit
# ======================================
st.set_page_config(
    page_title="AI Vision App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk dark mode elegan
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
    .stProgress > div > div > div {
        background-color: #00ccff;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================
# Load Models (Cache)
# ======================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ======================================
# Load Label Classes
# ======================================
# Bisa ubah sesuai file label kamu
label_dict = {
    0: "Kucing",
    1: "Anjing",
    2: "Burung",
    3: "Mobil",
    4: "Manusia"
}

# ======================================
# UI Utama
# ======================================
st.title("🧠 AI Vision App")
st.caption("Deteksi Objek (YOLO) & Klasifikasi Gambar (TensorFlow)")

menu = st.sidebar.radio("Pilih Mode:", ["🔍 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ======================================
# Fungsi Download
# ======================================
def get_downloadable_image(image_array):
    img_pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# ======================================
# Logika Utama
# ======================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Layout dua kolom
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(img, use_container_width=True)

    with col2:
        if menu == "🔍 Deteksi Objek (YOLO)":
            st.subheader("📦 Hasil Deteksi Objek")
            start_time = time.time()
            results = yolo_model(img)
            inference_time = time.time() - start_time
            result_img = results[0].plot()

            st.image(result_img, use_container_width=True)
            st.success(f"Waktu inferensi: {inference_time:.2f} detik")

            # Tombol download hasil deteksi
            st.download_button(
                label="💾 Unduh Hasil Deteksi",
                data=get_downloadable_image(result_img),
                file_name="hasil_deteksi.png",
                mime="image/png"
            )

        elif menu == "🧩 Klasifikasi Gambar":
            st.subheader("📊 Hasil Klasifikasi Gambar")

            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            start_time = time.time()
            prediction = classifier.predict(img_array)
            inference_time = time.time() - start_time

            class_index = np.argmax(prediction)
            class_label = label_dict.get(class_index, f"Class {class_index}")
            confidence = float(np.max(prediction))

            st.markdown(f"### 🏷️ Kelas: **{class_label}**")
            st.markdown(f"**Probabilitas:** {confidence*100:.2f}%")
            st.progress(confidence)
            st.success(f"Waktu inferensi: {inference_time:.2f} detik")

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit, YOLO & TensorFlow</p>", unsafe_allow_html=True)
