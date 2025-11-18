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
5. 
# Example Usage
python
from lrp_keras import lrp
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load Brain Tumor dataset
X_train, y_train = load_dataset(train_dir, img_size)
X_test,  y_test  = load_dataset(test_dir, img_size)

# Preprocess data
X_train_flat = X_train.reshape((2870, -1))  # flatten to (2870, 150528)
X_test_flat = X_test.reshape((394, -1))    # flatten to (394, 150528)

# Build a simple Keras model
model1 = keras.Sequential([
    layers.Dense(256, input_shape=(150528,), name='dense_17'),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])


model1.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Apply LRP to the model
# For output-to-second-last layer (layer 3→2)
z_k = epsilon + np.dot(A2, W3)  # shape: (4,)
s_k = R3 / z_k                  # shape: (4,)

c_j = np.dot(W3, s_k)           # shape: (128,)
R2 = A2 * c_j                   # shape: (128,)

# For second to first hidden (layer 2→1)
z_k = epsilon + np.dot(A1, W2)
s_k = R2 / z_k
c_j = np.dot(W2, s_k)
R1 = A1 * c_j

# For first hidden to input (layer 1→0)
z_k = epsilon + np.dot(A0, W1)
s_k = R1 / z_k
c_j = np.dot(W1, s_k)
R0 = A0 * c_j

# Visualize the relevance heatmaps
img_shape = (224, 224, 3)
heatmap_norm = R0 / np.max(R0)
plt.imshow(heatmap_norm.reshape(img_shape), cmap='hot')
plt.colorbar()
plt.show()


# References
Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting and understanding deep neural networks. Digital Signal Processing, 73, 1-15. Link

innvestigate: A toolbox to investigate deep neural networks. GitHub
