# 🍃 Deep Learning-Based Mango Leaf Disease Classification 🥭


## Project Overview

This project presents a deep learning-based system for **automatic detection of mango diseases**.  
The system uses a **ConvTransformer-based neural network** trained on RGB images of mango leaves to classify them into different disease categories.

The main goal is to provide a **user-friendly interface for farmers, researchers, and agricultural experts** to identify diseases in mango leaves quickly and accurately.

##  Demo

[Demo Video](https://drive.google.com/file/d/10QR5rzo8KLBqiTV07sfAUKmm0Xh5ZL90/view?usp=sharing)

##  Disease Classes

The model can classify mango leaf images into the following categories:

1. Anthracnose  
2. Bacterial Black Spots
3. Powdery Mildew 
4. Nutritional Deficiency
5. Healthy

---

##  Features

- Automated mango disease detection  
- Deep learning-based model (ConvTransformer architecture)  
- Confidence score for predictions  
- Streamlit web app for real-time predictions  
- Supports multiple image uploads  
- Minimal and professional interface  

---

##  Dataset

The dataset consists of **RGB images of mango crop(all organs except roots)**, labelled according to disease type.  
- Images were preprocessed and resized to **224x224 pixels**  
- Data augmentation techniques were applied during training (rotation, flipping, etc.)  
- Split into training and validation sets  

⚠️ Note: You do not need to retrain the model to use this app. Pre-trained weights are included.


## Model Architecture

- Backbone: **ConvTransformer**  
- Input: RGB images (224x224)  
- Output: 5 classes  
- Framework: **PyTorch**  
- Model is trained using **CrossEntropyLoss** and optimized with **Adam**.  

---

## Installation

### Clone the repository:

```bash
git clone https://github.com/Parul05101991/Mango_Disease_Detection
cd mango-disease-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

- Upload one or more mango leaf images.
- The app predicts the disease class and shows the confidence score.

##  Folder Structure

```text

mango-disease-detection/
│
├── app.py                # Streamlit web application
├── prediction.py         # Model loading & inference
├── training.py           # Training pipeline
├── requirements.txt      # Dependencies
├── README.md
│
├── models/               # Model architectures
│   ├── base_model.py
│   └── conv_trans.py
│
├── datasets/             # Data transformations
│   └── mango.py
│
├── weights/              # Pre-trained model
│   └── best_model.pt
│
├── Test_images/          # Sample images for testing
```

##  Acknowledgements
- ConvTransformer architecture adapted for plant disease classification.
- Dataset and preprocessing methods are based on public mango leaf datasets.
- Built using PyTorch and Streamlit.
