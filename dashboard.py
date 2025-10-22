import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import time
import cv2

# ======================================
# Konfigurasi Tampilan Streamlit
# ======================================
st.set_page_config(
    page_title="Pendeteksi Ekspresi",
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
    yolo_model = YOLO("model/Cahyo_Laporan4.pt")  # Model deteksi wajah
    classifier = tf.keras.models.load_model("model/Cahyo_Laporan2 (1).h5")  # Model klasifikasi ekspresi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ======================================
# Label Ekspresi
# ======================================
label_dict = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Pain",
    5: "Sad"
}

# ======================================
# UI Utama
# ======================================
st.title("🧠 Pendeteksi Ekspresi Wajah")
st.caption("Deteksi Wajah (YOLO) & Klasifikasi Ekspresi (TensorFlow)")

menu = st.sidebar.radio("Pilih Mode:", ["🔍 Deteksi Wajah", "🧩 Klasifikasi Ekspresi"])
uploaded_file = st.file_uploader("Unggah Gambar Wajah", type=["jpg", "jpeg", "png"])

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
    img_cv2 = np.array(img)

    # Layout dua kolom
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(img, use_container_width=True)

    with col2:
        if menu == "🔍 Deteksi Wajah":
            st.subheader("📦 Hasil Deteksi Wajah")

            start_time = time.time()
            results = yolo_model(img_cv2)
            inference_time = time.time() - start_time

            # Ambil hasil deteksi (gambar + koordinat)
            result_img = results[0].plot()
            st.image(result_img, use_container_width=True)
            st.success(f"Waktu inferensi: {inference_time:.2f} detik")

            # Tombol download hasil deteksi
            st.download_button(
                label="💾 Unduh Hasil Deteksi",
                data=get_downloadable_image(result_img),
                file_name="hasil_deteksi_wajah.png",
                mime="image/png"
            )

            # Crop wajah dari hasil deteksi (kalau ingin digunakan)
            boxes = results[0].boxes.xyxy
            if len(boxes) > 0:
                st.markdown("### 🔍 Wajah Terdeteksi:")
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    face_crop = img_cv2[y1:y2, x1:x2]
                    face_pil = Image.fromarray(face_crop)
                    st.image(face_pil, caption=f"Wajah {i+1}", width=200)

        elif menu == "🧩 Klasifikasi Ekspresi":
            st.subheader("📊 Hasil Klasifikasi Ekspresi")

            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi ekspresi
            start_time = time.time()
            prediction = classifier.predict(img_array)
            inference_time = time.time() - start_time

            class_index = np.argmax(prediction)
            class_label = label_dict.get(class_index, f"Class {class_index}")
            confidence = float(np.max(prediction))

            st.markdown(f"### 🏷️ Ekspresi: **{class_label}**")
            st.markdown(f"**Probabilitas:** {confidence*100:.2f}%")
            st.progress(confidence)
            st.success(f"Waktu inferensi: {inference_time:.2f} detik")

# ======================================
# Footer
# ======================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit, YOLO & TensorFlow</p>",
    unsafe_allow_html=True
)
