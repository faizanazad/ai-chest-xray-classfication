# AI-Based Chest X-Ray Classification

An AI-based web application for chest X-ray image classification using a trained Convolutional Neural Network (CNN).

The application provides a simple web interface where users can upload an X-ray image and receive the model's predicted class.


## Live Demo [Open X-ray classification App]  (https://ai-chest-xray-classfication.onrender.com/)



## Features

- Chest X-ray image upload
- Image preview before prediction
- CNN-based image classification
- Image preprocessing
- FastAPI backend
- Jinja2 HTML templates
- Responsive web interface
- Model confidence display

## Tech Stack

- Python
- FastAPI
- TensorFlow / Keras
- NumPy
- Pillow
- Scikit-learn
- HTML
- CSS

📸 Project Screenshots

1. Home Page

"Home Page" (homepage.png)

2. Upload X-ray

"Upload X-ray" (upload-xray.png)

3. Prediction Result

"Prediction Result" (prediction-result.png)


## Project Structure

```text
x-ray-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── CNN_Covid19_Xray_version.h5
│   └── Lable_encoder.pkl
│
├── templates/
│   ├── index.html
│   ├── camera.html
│   └── result.html
│
└── static/
    └── style.css



