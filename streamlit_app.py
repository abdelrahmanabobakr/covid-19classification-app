import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import urllib.request
import os

st.title("Covid-19 Classification")
st.write("Covid-19 Prediction From X-ray")

# Download the model if it does not exist
MODEL_PATH = "covid_19_model.h5"
MODEL_URL = "https://github.com/abdelrahmanabobakr/covid-19classification-app/releases/download/v1.0/covid_19_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")

# Load the model
Model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = {0: "Covid", 1: "Normal", 2: "Viral Pneumonia"}

# File uploader for the image
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:  # Check if an image is uploaded
    # Open the uploaded image
    img = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image.')
    
    # Convert the image to a NumPy array
    new_image = np.array(img)
    
    # Resize the image to 224 x 224
    new_image = cv2.resize(new_image, (224, 224))
    
    # Convert the image to grayscale (optional if the image is colored)
    if new_image.ndim == 2:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
    
    # Normalize the image
    new_image = new_image.astype('float32') / 255.0
    
    # Reshape the image
    new_image = new_image.reshape(1, 224, 224, 3)
    
    # Make a prediction
    prediction = Model.predict(new_image)
    
    # Extract the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = CLASS_LABELS[predicted_class]
    
    # Display the result
    st.write(f'Predicted Class: **{predicted_label}**')
