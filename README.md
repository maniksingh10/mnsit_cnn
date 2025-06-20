# 🧠 Handwritten Digit Classification with CNN (MNIST)

This project builds and trains a **Convolutional Neural Network (CNN)** using the **MNIST dataset** to classify handwritten digits (0–9). It also includes a Streamlit app for deploying the model and visualizing predictions.

---

## 📁 Project Structure

```
mnist-cnn/
├── model.h5                 # Trained CNN model
├── app.py                   # Streamlit app to test predictions
├── requirements.txt         # Required packages
└── README.md                # Project documentation
```

---

## 📊 Dataset

- **Source**: MNIST Dataset
- **Shape**: 60,000 training images and 10,000 test images
- **Image Size**: 28x28 pixels, grayscale

---

## 🏗️ Model Architecture

- **Input Shape**: (28, 28, 1)
- **Layers**:
  - Conv2D(32 filters) → ReLU → MaxPooling
  - Conv2D(64 filters) → ReLU → MaxPooling
  - Dropout (0.5)
  - Flatten
  - Dense(100) → ReLU
  - Dropout (0.5)
  - Dense(10) → Softmax (for classification)

---

## 📈 Training

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 128
- **Validation**: On MNIST test data

---

## 📊 Evaluation

- **Final Accuracy**: ~98% (varies per run)
- **Validation Graphs**: Accuracy vs Epoch shown using matplotlib

---

## 🚀 Streamlit App

You can use the `app.py` file to:
- Upload an image or choose a test sample
- View the predicted and actual digit

Run the app:
```bash
streamlit run app.py
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## 📌 Requirements

```
streamlit
tensorflow==2.15.0
pillow
numpy
```


