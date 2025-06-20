import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("mnist_model.keras")

# App title
st.title("🧠 Handwritten Digit Recognition (MNIST CNN)")
st.write("Upload a 28x28 grayscale image of a digit (0–9) to predict.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', width=150)

    # Resize to 28x28
    image = ImageOps.invert(image)  # Invert if background is white
    image = image.resize((28, 28))
    
    # Convert image to numpy
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.success(f"Predicted Digit: **{predicted_class}**")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
