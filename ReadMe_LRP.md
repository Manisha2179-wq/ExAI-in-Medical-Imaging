# LRP Implementation in Keras
This repository contains an implementation of the Layer-wise Relevance Propagation (LRP) algorithm for Keras-based neural networks, enabling explanation and visualization of model predictions via relevance heatmaps.

# Table of Contents
 1. Introduction
 2. Installation
 3. Usage
 4. Example Usage
 5. References
 
# Introduction
Layer-wise Relevance Propagation (LRP) is a technique to explain neural network decisions by attributing relevance scores to each input feature, decomposing predictions back through the layers.
This implementation supports Keras models and allows visualizing relevance heatmaps that indicate which parts of the input contributed most to the model's output.

# Installation
  Clone the repository:
  git clone https://github.com/your-username/lrp-keras.git

  Install the required dependencies:
  pip install -r requirements.txt


# Usage
1. Import the necessary modules:
python
from lrp_keras import lrp
from keras.models import load_model

2. Load your trained Keras model:
python
model = load_model('path/to/your/model.h5')

4. Apply LRP to the model and get the relevance scores:
python
X = ... # Your input data
relevance = lrp(model, X)

6. Visualize the relevance heatmaps using your preferred visualization library (e.g., Matplotlib).

# Example Usage
import matplotlib.pyplot as plt
from lrp_keras import lrp
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

1. Load your dataset
X_train, y_train = load_dataset(train_dir, img_size)
X_test, y_test = load_dataset(test_dir, img_size)

2. Flatten input if needed
X_train_flat = X_train.reshape((2870, -1))
X_test_flat = X_test.reshape((394, -1))

3. Define simple Keras model
model1 = Sequential([
    Dense(256, input_shape=(150528,), name='dense_1'),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model1.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

4. Train your model before LRP usage
model1.fit(X_train_flat, y_train, epochs=..., validation_data=(X_test_flat, y_test))

5. Layer-wise Relevance Propagation backward pass (example for 3 layers)
epsilon = 1e-9

6. For output-to-second-last layer (layer 3→2)
z_k = epsilon + np.dot(A2, W3)  # shape: (4,)
s_k = R3 / z_k                  # shape: (4,)

c_j = np.dot(W3, s_k)           # shape: (128,)
R2 = A2 * c_j                   # shape: (128,)

7. For second to first hidden (layer 2→1)
z_k = epsilon + np.dot(A1, W2)
s_k = R2 / z_k
c_j = np.dot(W2, s_k)
R1 = A1 * c_j

8. For first hidden to input (layer 1→0)
z_k = epsilon + np.dot(A0, W1)
s_k = R1 / z_k
c_j = np.dot(W1, s_k)
R0 = A0 * c_j

9. Visualization of normalized heatmap
heatmap_norm = relevance / np.max(relevance)
plt.imshow(heatmap_norm.reshape((224, 224, 3)), cmap='hot')
plt.colorbar()
plt.show()

# References
1. Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting and understanding deep neural networks. Digital Signal Processing, 73, 1-15.
2. innvestigate: A toolbox to investigate deep neural networks. GitHub.

This README approach provides clarity on installation, usage, and example code for LRP with Keras models, structured to help users understand and apply LRP effectively.
