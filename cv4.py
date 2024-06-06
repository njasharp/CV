import cv2
import streamlit as st
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a Streamlit app
st.title("Webcam App")
st.sidebar.title("Webcam Menu")

# Create a button to capture an image
capture_button = st.sidebar.button("Capture Image", key="capture_button")

# Create a canvas to display the image
import cv2
import streamlit as st
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a Streamlit app
st.title("Webcam App")
st.sidebar.title("Webcam Menu")

# Create a button to capture an image
capture_button = st.sidebar.button("Capture Image", key="capture_button")

# Create a reset button
reset_button = st.sidebar.button("Reset", key="reset_button")

# Create a canvas to display the image
canvas = st.empty()
captured_image_placeholder = st.sidebar.empty()

# Function to reset the capture button state
def reset():
    st.experimental_rerun()

if reset_button:
    reset()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        st.error("Failed to capture image from webcam")
        break

    # Convert the frame to a numpy array
    frame = np.array(frame)

    # Display the frame in the canvas
    canvas.image(frame, channels="BGR")

    # Check if the capture button was clicked
    if capture_button:
        # Capture the image
        ret, frame = cap.read()
        frame = np.array(frame)
        captured_image_placeholder.image(frame, caption="Captured Image", channels="BGR")

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

        # Reset the capture button
        capture_button = False

    # Exit the loop if the user closes the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()