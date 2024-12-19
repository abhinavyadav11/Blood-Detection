import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# Set page title and layout
st.set_page_config(page_title="Blood Component Detection", page_icon="🩸", layout="wide")

# Styling the app
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f6f9;
        }
        .title {
            color: #000;
            font-size: 40px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .subtitle {
            color: #555;
            font-size: 20px;
            text-align: center;
        }
        .upload-container {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .predict-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 18px;
            width: 100%;
            cursor: pointer;
        }
        .predict-btn:hover {
            background-color: #45a049;
        }
        .stImage {
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Set title and subtitle
st.markdown('<div class="title">Blood Component Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect and classify blood cells in uploaded images</div>', unsafe_allow_html=True)

# Load the YOLO model
model = YOLO('/Users/abhinavyadav/Downloads/final_best(blood_detection).pt')

# Image uploader with custom style
with st.container():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload Image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prediction button
    with st.container():
        if st.button('Classify Image', key="predict", help="Click to start classification", use_container_width=True):
            st.write("Classifying... Please wait.")

            # Display a progress bar while the model processes
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i+1)

            # Convert the image for prediction
            ### img_array = np.array(image)

            # Perform prediction with a low confidence threshold
            results = model.predict(source=image, imgsz=640, conf=0.1)

            # Display results
            for result in results:
                print(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)
                # Print bounding boxes, class IDs, and confidence

            # Show results
            for i in range (1, 25):
                if len(result[0].boxes) > 0:
                    # Extract detection results
                    st.write("Bounding Boxes:", result[0].boxes.xyxy)
                    st.write("Class IDs:", result[0].boxes.cls)
                    st.write("Confidence Scores:", result[0].boxes.conf)

                    # Plot the results (image with bounding boxes)
                    output_image = result[i].plot()  # This should plot bounding boxes
                    st.image(output_image, caption="Predicted Image with Bounding Boxes", use_container_width=True)
                else:
                    st.write("No objects detected.")
                    st.write("Raw Prediction Results:", results)
                    st.write("Boxes:", results[0].boxes.xyxy)
                    st.write("Classes:", results[0].boxes.cls)
                    st.write("Confidences:", results[0].boxes.conf)




