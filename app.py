import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("/home/t-i/Hackathom/diabetic_retinopathy_transformer_balanced.h5")
    return model

model = load_model()

# Class labels
CLASS_NAMES = ["No DR (Healthy)", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Streamlit UI
st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal image to classify its severity.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Show result
    st.subheader(f"Prediction: {CLASS_NAMES[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
