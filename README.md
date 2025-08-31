# 🔢 ML Digit Recognition — Handwritten Digit Classifier

A Python-based machine learning project that trains, evaluates, and applies a handwritten digit classifier using **HOG (Histogram of Oriented Gradients)** features and **scikit-learn**.

---

## 🎯 Project Overview

- **trainer.py** — Trains a classifier on digit images using HOG features and generates a pickle model (`mnist_sklearn_hog_pipeline.pkl`)
- **hog_utils.py** — Contains helper functions for extracting HOG features from images
- **writer.py** — Handles writing/reading data (dataset manipulation or custom image input)
- The trained model can recognize handwritten digits 0–9 with high accuracy

---

## ⚡ Prerequisites

Ensure you have **Python 3.8+** installed.

Install required Python libraries:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Train the Model

```bash
python trainer.py
```

This trains the classifier and saves the model to `mnist_sklearn_hog_pipeline.pkl`.

### 2. Make Predictions

Add a small script or snippet to load the model and make predictions:

```python
import pickle
from hog_utils import extract_hog_features

# Load the trained model
with open("mnist_sklearn_hog_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Example: load a digit image, extract features, predict
# features = extract_hog_features(image)
# print("Predicted digit:", model.predict([features])[0])
```

### 3. Extend Functionality
- Integrate a simple **GUI (Tkinter)** to draw digits and display predictions in real time
- Add script for **batch prediction** on handwritten image folders
- Create a **web interface** for live demonstrations

---

## 💡 Why This Project?

- Demonstrates **computer vision fundamentals** — feature extraction (HOG) + ML pipeline
- Clean separation of utilities (`hog_utils.py`), training (`trainer.py`), and IO (`writer.py`)
- **Portable and lightweight** — ideal for quick demos, portfolio showcases, and Python skill validation

---

## 🔮 Next Steps & Ideas

- Add **web interface** with Flask to create a live demo (deployable via Render or Heroku)
- Use **deep learning (CNN)** for improved accuracy
- Expand dataset to include digits with noise, different handwriting styles — test your model's robustness
- Document performance metrics (accuracy, confusion matrix) clearly in a report or Jupyter Notebook

---

## 📁 Project Structure

```
ml-digit-recognition/
├── hog_utils.py                      # HOG feature extraction utilities
├── trainer.py                       # Model training script
├── writer.py                        # Data I/O handling
├── mnist_sklearn_hog_pipeline.pkl   # Trained model (generated)
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

---

## 👨‍💻 Author

**Yaswanth V**  
B.E. CSE (AI & ML) | TCE  

📧 **Email:** vsyaswanth008@gmail.com  
🔗 **GitHub:** [https://github.com/Yaswanth876](https://github.com/Yaswanth876)

---

⭐ **If you found this project helpful, please give it a star!**
