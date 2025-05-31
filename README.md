# Flower Prediction using Convolutional Neural Network

A deep learning project that uses Convolutional Neural Networks (CNN) to classify and predict different flower species from images.

## 🌸 Project Overview

This project implements a CNN model to automatically identify and classify flower species from input images. The model is trained to recognize multiple flower types and can predict the species of new flower images with high accuracy.

## 🚀 Features

- **Multi-class flower classification** using deep learning
- **Convolutional Neural Network** architecture optimized for image recognition
- **Image preprocessing** pipeline for data augmentation and normalization
- **Model evaluation** with accuracy metrics and confusion matrix
- **Prediction capability** for new flower images
- **Visualization** of training progress and model performance

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **OpenCV** - Image processing
- **Scikit-learn** - Model evaluation metrics
- **Pandas** - Data manipulation

## 📊 Dataset

The model is trained on a flower dataset containing images of various flower species such as:
- Roses
- Daisies
- Dandelions
- Sunflowers
- Tulips
- And other flower varieties

*Note: Please ensure you have the appropriate dataset downloaded and placed in the correct directory structure.*

## 🏗️ Model Architecture

The CNN model consists of:
- **Convolutional layers** for feature extraction
- **MaxPooling layers** for dimensionality reduction
- **Dropout layers** for preventing overfitting
- **Dense layers** for classification
- **Softmax activation** for multi-class prediction

## 📋 Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
opencv-python>=4.6.0
scikit-learn>=1.1.0
pandas>=1.4.0
pillow>=9.0.0
```
- **Trained model link:** https://drive.google.com/file/d/16jebQn4TCy_ov_FLzMXeii6FMiP5ipLl/view?usp=sharing
- **Flowers Training and Validation data link:** https://drive.google.com/file/d/1opXD0iREqT-ZbwObYUeWJf5zrH7aufcs/view?usp=sharing

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dhruvil03/Convolutional_Neural_Network.git
   cd Convolutional_Neural_Network/Flower_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   - Ensure your flower images are organized in appropriate folders
   - Update dataset paths in the configuration files

4. **Train the model**
   ```bash
   python train_model.py
   ```

5. **Make predictions**
   ```bash
   python predict.py --image_path path/to/your/flower/image.jpg
   ```

## 📈 Model Performance

- **Training Accuracy**: [Add your achieved accuracy]
- **Validation Accuracy**: [Add your achieved accuracy]
- **Test Accuracy**: [Add your achieved accuracy]
- **Loss**: [Add final loss value]

## 📁 Project Structure

```
Flower_prediction/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   └── saved_model/
├── notebooks/
│   └── flower_classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── utils/
│   └── visualization.py
├── requirements.txt
└── README.md
```

## 🔍 Key Features

### Data Preprocessing
- Image resizing and normalization
- Data augmentation (rotation, flipping, zoom)
- Train-validation-test split

### Model Training
- Custom CNN architecture
- Transfer learning options
- Early stopping and model checkpointing
- Learning rate scheduling

### Evaluation
- Accuracy and loss plotting
- Confusion matrix generation
- Classification report
- Model performance visualization

## 📸 Sample Predictions

The model can classify flowers into different categories with confidence scores:

```
Input: sunflower_image.jpg
Prediction: Sunflower (98.5% confidence)

Input: rose_image.jpg
Prediction: Rose (96.2% confidence)
```
# Screenshot
![WhatsApp Image 2024-02-29 at 5 00 07 PM](https://github.com/Dhruvil03/Convolutional_Neural_Network/assets/90698427/097e316d-05b2-44c7-8c5d-38c85ca16fb4)


⭐ **If you found this project helpful, please give it a star!** ⭐
