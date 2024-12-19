# Blood Component Detection

## Overview

The Blood Component Detection project aims to classify and localize different blood components (e.g., red blood cells, white blood cells, platelets) in an uploaded image. Using a fine-tuned YOLO (You Only Look Once) model, the app predicts bounding boxes, class IDs, and confidence scores for detected components. The tool is designed for educational purposes and demonstrates how to use deep learning models in real-world applications.

# Features

- User-friendly interface built with Streamlit.

- Upload support for image formats: JPG, JPEG, PNG.

- Detection and classification of blood components with bounding box visualization.

- Real-time feedback and progress visualization.

- Custom styling to enhance user experience.

# Technical Details

### 1. Frontend:

- Framework: Streamlit

- Styling: Custom HTML and CSS for a modern and intuitive design.

- Dynamic UI Components:

    - File uploader for image inputs.

    - Prediction button to initiate detection.

    - Progress bar for real-time feedback.

### 2. Backend:

- Deep Learning Framework: PyTorch (via Ultralytics YOLO library).

- Model: Pretrained YOLO model fine-tuned for blood component detection.

- Libraries Used:

    - numpy for array operations.

    - Pillow for image processing.

    - ultralytics for YOLO inference.

### 3. Deployment Environment:

- Local environment: Python 3.10 virtual environment (.venv).

- Deployment-ready code for hosting on platforms like Streamlit Cloud or Heroku.

### 4. Key Functionalities:

- Image Preprocessing: Converts images to RGB format for compatibility.

- Model Prediction: Performs detection with a confidence threshold of 0.1 to capture all possible components.

- Result Visualization: Outputs bounding boxes on the uploaded image and displays detection results (e.g., class labels and confidence scores).

## Installation


#### - Clone the Repository:


git clone https://github.com/username/BloodCellsDetection.git
cd BloodCellsDetection


#### - Create a Virtual Environment:

python3 -m venv .venv
source .venv/bin/activate 

#### - Install Dependencies:

pip install -r requirements.txt

#### - Download the YOLO Model:
Place the fine-tuned YOLO weights (blood_detection_model.pt) in the project directory. Update the path in the code if necessary.


## Usage

#### - Run the Streamlit App:

streamlit run app.py

- Upload an Image: Use the file uploader to select a blood sample image.

- Classify the Image: Click the "Classify Image" button to start detection.

    - View Results:

    - Bounding boxes for detected components.

    - Predicted class labels and confidence scores.

## Model Details

## Architecture: YOLO

- Pretrained Weights: YOLO model trained on a custom dataset of blood components.

- Fine-tuning:

- Dataset: Blood component dataset with bounding box annotations.

- Augmentation techniques: Horizontal flipping, scaling, and color jittering.

    - Training Epochs: 50

    - Optimizer: AdamW

    - Learning Rate: 1e-4

## Future Enhancements

- Add support for more blood components and conditions.

- Implement real-time detection using video inputs.

- Deploy the app on Streamlit Cloud or AWS for broader accessibility.

- Improve model accuracy with additional dataset augmentation and hyperparameter tuning.

- Include interpretability tools like Grad-CAM to visualize model focus areas.

## Acknowledgments

- Ultralytics YOLO: For providing the state-of-the-art detection framework.

- Streamlit: For enabling rapid development of interactive web apps.

- Pillow & NumPy: For image preprocessing and numerical computations.