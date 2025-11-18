# LRP Implementation in Keras
This repository contains an implementation of the Layer-wise Relevance Propagation (LRP) algorithm for neural networks built using the Keras deep learning library.
# Table of Contents
 Introduction
 Installation
 Usage
 Example Usage
 References
# Introduction
LRP is a technique for explaining the predictions of neural networks by decomposing the prediction into relevance scores for each input feature. This implementation allows you to apply LRP to Keras models and visualize the relevance heatmaps.
The code is based on the work of Montavon et al. and the innvestigate library.
# Installation
Clone the repository:
bash
git clone https://github.com/your-username/lrp-keras.git
Install the required dependencies:
bash
pip install -r requirements.txt
# Usage
1. Import the necessary modules:
python
from lrp_keras import lrp
from keras.models import load_model
2. Load your trained Keras model:
python
model = load_model('path/to/your/model.h5')
3. Apply LRP to the model and get the relevance scores:
python
X = ... # Your input data
relevance = lrp(model, X)
4. Visualize the relevance heatmaps using your preferred visualization library (e.g., Matplotlib).
# Example Usage
python
from lrp_keras import lrp
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_test = X_test.reshape(-1, 28*28) / 255.0
# Build a simple Keras model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Apply LRP to the model
relevance = lrp(model, X_test[:10])

# Visualize the relevance heatmaps
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.imshow(relevance[i].reshape(28, 28), cmap='jet', alpha=0.5)
    plt.axis('off')
plt.show()

# References
Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting and understanding deep neural networks. Digital Signal Processing, 73, 1-15. Link
innvestigate: A toolbox to investigate deep neural networks. GitHub