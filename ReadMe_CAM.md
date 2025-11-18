# CAM Implementation in Keras
This repository contains an implementation of Class Activation Mapping (CAM) using Keras. CAM is a technique used to visualize the regions in an image that are important for a model's predictions, providing insights into the decision-making process of convolutional neural networks (CNNs).

# Table of Contents
1. Introduction
2. Installation
3. Usage: Example Usage
4. References

# Introduction
Class Activation Mapping (CAM) allows users to understand which parts of an image contribute most to the predictions made by CNNs. By generating heatmaps, CAM helps interpret model decisions and can be particularly useful for debugging and improving model performance.

# Installation
1. Clone the repository:
git clone https://github.com/your-username/cam-keras.git
cd cam-keras
2. Install the required dependencies:
pip install -r requirements.txt

# Usage
1. Import the necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception, imagenet_utils

2. Load your pre-trained model (e.g., Xception):
model = xception.Xception(weights='imagenet')

3. Define a function to preprocess the input image:
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return xception.preprocess_input(array)

4. Create the CAM heatmap:
def make_cam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

5. Visualize the results:
def display_cam(img_path, heatmap):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.jet(np.arange(256))[:, :3]
    jet_heatmap = jet[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    plt.imshow(superimposed_img / 255.)
    plt.axis('off')
    plt.show()

img_path = 'path/to/your/image.jpg'
img_array = get_img_array(img_path, size=(299, 299))
predicted_class_index = np.argmax(model.predict(img_array))

heatmap = make_cam_heatmap(img_array, model, 'block14_sepconv2_act', pred_index=predicted_class_index)
display_cam(img_path, heatmap)

# Example Usage
You can find an example usage in the CAM.ipynb file provided in this repository.

# References
Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). "Learning Deep Features for Discriminative Localization." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Link to paper.
