import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import joblib
import os
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🧠", layout="centered")

st.title("🧠 Skin Cancer Detection System (Multimodal AI)")
st.write("Upload an image and enter patient details for prediction.")

st.warning(
    "This tool is for educational/research use only and is not a medical diagnosis. "
    "Please consult a qualified dermatologist for medical advice."
)

MODEL_PATH = "multimodal_model.keras"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error("Failed to load model or scaler.")
    st.exception(e)
    st.stop()

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

age = st.number_input("Enter Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Select Sex", ["Male", "Female", "Unknown"])
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])
show_debug = st.checkbox("Show debug information")

if st.button("Predict"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((128, 128))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        scaled_age = scaler.transform(
            np.array([[age]], dtype=np.float32)
        ).flatten()[0]

        sex_female_one_hot = 1.0 if sex == "Female" else 0.0
        sex_male_one_hot = 1.0 if sex == "Male" else 0.0
        sex_unknown_one_hot = 1.0 if sex == "Unknown" else 0.0

        metadata = np.array([[
            scaled_age,
            sex_female_one_hot,
            sex_male_one_hot,
            sex_unknown_one_hot
        ]], dtype=np.float32)

        if show_debug:
            st.write("Image shape:", img.shape)
            st.write("Metadata shape:", metadata.shape)
            st.write("Model input shape:", model.input_shape)

        prediction = model.predict([img, metadata], verbose=0)

        if show_debug:
            st.write("Prediction shape:", prediction.shape)
            st.write("Raw prediction:", prediction)

        predicted_index = int(np.argmax(prediction[0]))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[0][predicted_index] * 100)

        st.subheader("Result")
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.write("Class probabilities:")
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Confidence (%)": [float(prediction[0][i] * 100) for i in range(len(class_names))],
        }).sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
