# AI-Based Chest X-Ray Classification

An AI-based web application for chest X-ray image classification using a trained Convolutional Neural Network (CNN).

The application provides a simple web interface where users can upload an X-ray image and receive the model's predicted class.

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