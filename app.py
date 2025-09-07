import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import time
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Klasifikasi Gambar Anjing vs Kucing",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🐶🐱 Klasifikasi Gambar: Anjing vs Kucing")
st.write(
    """
    Selamat datang di aplikasi klasifikasi gambar!  
    Aplikasi ini menggunakan **deep learning** untuk menentukan apakah gambar yang Anda unggah adalah **anjing** atau **kucing**.
    
    👉 Cukup unggah gambar Anda, lalu biarkan model bekerja.
    """
)
st.markdown("---")

# --- Memuat Model dengan caching ---
# --- Memuat Model dengan caching ---
@st.cache_resource
def load_model():
    from tensorflow.keras.layers import Flatten

    with st.spinner('🔄 Model sedang diunduh dan dimuat...'):
        model_path = hf_hub_download(
            repo_id="benJowl/KlasifikasiAnjingKucing",
            filename="best_transfer_model.h5"
        )
        # Tambahkan custom_objects supaya Flatten lama dikenali
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"Flatten": Flatten}
        )
        st.success('✅ Model berhasil dimuat!')
        return model

model = load_model()
st.write("Ukuran input model:", model.input_shape)

# --- Fungsi Pra-pemrosesan Gambar ---
def preprocess_image(img):
    """Mengubah ukuran dan menormalisasi gambar agar sesuai dengan input model."""
    target_size = model.input_shape[1:3]
    img = img.resize(target_size)  
    img_array = np.array(img)      
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

# --- Uploader File ---
uploaded_file = st.file_uploader(
    "📂 Unggah gambar anjing atau kucing di sini",
    type=['jpg', 'png', 'jpeg']
)

if uploaded_file:
    # --- Menampilkan Gambar yang Diunggah ---
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang diunggah", use_column_width=True)

    # --- Prediksi Gambar ---
    st.markdown("---")

    with st.spinner('🤖 Model sedang memprediksi...'):
        input_img = preprocess_image(img)
        pred = model.predict(input_img)
        time.sleep(1)  # jeda biar spinner kelihatan

    # --- Menampilkan Hasil Prediksi ---
    if pred.shape[1] == 1:  # model biner
        confidence = float(pred[0][0])
        label = "Anjing" if confidence > 0.5 else "Kucing"
        confidence_display = confidence if label == "Anjing" else 1 - confidence
        # Buat distribusi probabilitas untuk chart
        probas = {
            "Anjing": confidence,
            "Kucing": 1 - confidence
        }
    else:  # model multi-kelas
        confidence = float(np.max(pred[0]))
        predicted_class_index = np.argmax(pred[0])
        label = "Anjing" if predicted_class_index == 0 else "Kucing"
        confidence_display = confidence
        # Buat distribusi probabilitas untuk chart
        probas = {
            "Anjing": float(pred[0][0]),
            "Kucing": float(pred[0][1])
        }

    st.subheader("📊 Hasil Prediksi")
    st.metric(
        label="Prediksi Gambar ini adalah:",
        value=f"{label} {'🐶' if label == 'Anjing' else '🐱'}",
        delta=f"Keakuratan: {confidence_display * 100:.2f}%" 
    )

    # --- Bar Chart Probabilitas ---
    st.markdown("### 🔍 Probabilitas Detail")
    prob_df = pd.DataFrame(
        {"Kelas": list(probas.keys()), "Probabilitas": list(probas.values())}
    )
    st.bar_chart(prob_df.set_index("Kelas"))

    # --- Expander untuk Detail Teknis ---
    with st.expander("⚙️ Lihat Detail Teknis"):
        st.write("Hasil prediksi mentah dari model:")
        st.json(pred.tolist())
        st.write("Bentuk (shape) input untuk prediksi:")
        st.write(input_img.shape)
        st.write("Nilai ambang batas (threshold): 0.5 (untuk model biner)")
