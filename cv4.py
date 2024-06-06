import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Create a Streamlit app
st.title("Webcam App")
#st.sidebar.title("Webcam Menu")

# Create a button to reset the image
#reset_button = st.sidebar.button("Reset", key="reset_button")

# Create a placeholder for the webcam input
uploaded_image = st.camera_input("Capture Image")

# Function to reset the captured image
if reset_button:
    st.experimental_rerun()

if uploaded_image:
    # Convert the uploaded image to an OpenCV format
    img = Image.open(uploaded_image)
    frame = np.array(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the final output
    st.image(frame, caption="Processed Image with Face Bounding Box", channels="BGR")
