import streamlit as st
import cv2
import numpy as np
from PIL import Image
import traceback

st.title("Face Detection App")

# Sidebar option to upload an image file
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        img = Image.open(uploaded_file)
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
        st.image(frame[:, :, ::-1], caption="Processed Image with Face Bounding Box")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write(traceback.format_exc())
else:
    st.sidebar.write("Please upload an image to start face detection.")
