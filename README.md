# Chest X-Ray Detector 🩻

A deep learning application for classifying chest X-ray images into four categories: **COVID, Normal, Lung Opacity, and Viral Pneumonia**.

The project uses **Transfer Learning with MobileNetV2**, pretrained on ImageNet, and provides an interactive **Streamlit** web application for image prediction.

---

## 📌 Overview

Chest X-Ray Detector is a computer vision project designed to demonstrate how deep learning can be applied to medical image classification.

A pretrained **MobileNetV2** model is used as the feature extractor. The original classification layer is replaced with custom layers designed to classify chest X-ray images into four classes.

After training, the model is saved and integrated into a Streamlit application. Users can upload a chest X-ray image and receive:

- The predicted class
- Prediction confidence
- Probability for each class

> **Disclaimer:** This project is developed for educational purposes only. It is not intended to replace professional medical diagnosis or clinical decision-making.

---

## 🧠 Model

The project uses **MobileNetV2 with Transfer Learning**.

### Model Configuration

| Configuration      | Value                           |
| ------------------ | ------------------------------- |
| Architecture       | MobileNetV2                     |
| Pretrained Weights | ImageNet                        |
| Input Size         | 224 × 224 × 3                   |
| Number of Classes  | 4                               |
| Framework          | TensorFlow / Keras              |
| Classification     | Softmax                         |
| Optimizer          | Adam                            |
| Loss Function      | Sparse Categorical Crossentropy |

The pretrained MobileNetV2 base is initially frozen, allowing the newly added classification layers to learn the features required for the chest X-ray classification task.

### Classification Classes

The model predicts one of the following:

1. **COVID**
2. **Normal**
3. **Lung Opacity**
4. **Viral Pneumonia**

---

## 📊 Dataset

The project uses the **COVID-19 Radiography Database** containing chest X-ray images from the four target categories.

The images are:

- Converted to RGB
- Resized to **224 × 224**
- Preprocessed using MobileNetV2 preprocessing
- Split into training and testing sets

The dataset is not included in this repository because of its size and licensing considerations.

---

## 🖥️ Streamlit Application

The trained model is integrated into a web interface using **Streamlit**.

The application allows the user to:

1. Upload a chest X-ray image.
2. Preview the uploaded image.
3. Run the prediction.
4. View the predicted class.
5. View the confidence score.
6. View the probabilities for all four classes.

---

## 📂 Project Structure

```text
Chest_X-Ray_Detector/
│
├── app.py
├── model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

### Files

**`app.py`**
Contains the Streamlit application and the image prediction pipeline.

**`model.keras`**
The trained TensorFlow/Keras MobileNetV2 model.

**`requirements.txt`**
Contains the Python libraries required to run the project.

**`.gitignore`**
Specifies files and folders that should not be uploaded to GitHub.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ahmedosamasalem/Chest_X-Ray_Detector.git
```

Navigate to the project directory:

```bash
cd Chest_X-Ray_Detector
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL where the application can be opened in a web browser.

---

## 🔮 Prediction Workflow

```text
Chest X-Ray Image
        ↓
Image Upload
        ↓
Resize to 224 × 224
        ↓
Image Preprocessing
        ↓
MobileNetV2
        ↓
Feature Extraction
        ↓
Classification Layers
        ↓
Softmax Probabilities
        ↓
Predicted Class + Confidence
```

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2**
- **NumPy**
- **OpenCV**
- **Pillow**
- **Scikit-learn**
- **Streamlit**
- **Google Colab**
- **Kaggle**

---

## 🎯 Project Goals

This project demonstrates practical implementation of:

- Deep Learning
- Transfer Learning
- Image Preprocessing
- Multi-class Image Classification
- Model Evaluation
- TensorFlow/Keras
- Streamlit Deployment
