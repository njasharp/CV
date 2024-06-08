import streamlit as st
import cv2
import numpy as np
from PIL import Image
import traceback
import os

st.title("Face Detection App")

# Check if it's the first use of the app
if "first_use" not in st.session_state:
    st.session_state.first_use = True

# Sidebar option to use default image or upload an image
use_default_image = st.sidebar.checkbox("Use Default Image")

# If it's the first use or the checkbox is checked, use the default image
if use_default_image or st.session_state.first_use:
    default_image_path = "image.png"
    if os.path.exists(default_image_path):
        uploaded_file = open(default_image_path, "rb")
    else:
        st.error("Default image not found. Please make sure the default image 'image.png' is in the correct directory.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        img = Image.open(uploaded_file)
        frame = np.array(img)

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Provide the full path to the haarcascade XML file
        cascades_dir = '/path/to/your/haarcascades/directory/'
        face_cascade = cv2.CascadeClassifier(cascades_dir + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the final output
        st.image(frame[:, :, ::-1], caption="Processed Image with Face Bounding Box")

        # Update the first_use flag
        st.session_state.first_use = False

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write(traceback.format_exc())
else:
    st.sidebar.write("Please upload an image to start face detection.")
