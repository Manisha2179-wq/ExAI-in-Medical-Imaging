# CAM Implementation in Keras
This repository provides implementations of Class Activation Mapping (CAM) and Gradient-weighted Class Activation Mapping (Grad-CAM) for explaining the predictions of convolutional neural networks (CNNs). These techniques generate visual explanations by highlighting important regions in input images that are responsible for the model's decision.

# Table of Contents
1. Introduction
2. Installation
3. Usage: Example Usage
4. References

# Introduction
$1$. Class Activation Mapping (CAM):
CAM uses the output of the last convolutional layer combined with the weights of the final classification layer to produce a coarse heatmap highlighting class-specific discriminative image regions.

$2$. Gradient-weighted CAM (Grad-CAM):
Grad-CAM generalizes CAM to a wider variety of CNN architectures by using the gradients of the target class flowing into the last convolutional layer to produce the localization heatmap. Grad-CAM is popular for its applicability to many CNNs without architecture changes or retraining.

These methods enhance model interpretability and assist in understanding model failure modes, trust, and dataset bias.

# Installation
1. Clone the repository:
git clone https://github.com/your-username/cam-keras.git
cd cam-keras
2. Install the required dependencies:
pip install -r requirements.txt

# Usage
1. Import the necessary libraries:
from cam import CAM
from gradcam import GradCAM

2. Load your pre-trained model (e.g., VGG16):
from tensorflow.keras.applications import VGG16
model = VGG16(weights='imagenet')

3. Initialize CAM or Grad-CAM for your target layer:
   cam = CAM(model, target_layer='block5_conv3')         # For CAM
gradcam = GradCAM(model, target_layer='block5_conv3') # For Grad-CAM

4. Compute the heatmap for a given input:
   heatmap = gradcam.compute_heatmap(image, class_idx=target_class_idx)
   
5. Overlay and visualize the heatmap:
import cv2
import matplotlib.pyplot as plt

heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
overlay = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

plt.imshow(cv2.addWeighted(image, 0.6, overlay, 0.4, 0))
plt.axis('off')
plt.show()


# Example Usage
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

heatmap = gradcam.compute_heatmap(x, class_idx=None)  # None uses top predicted class


# References
1. Zhou et al., "Learning Deep Features for Discriminative Localization," CVPR 2016 (CAM)
2. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017
3. GitHub: Grad-CAM repository

This README provides users a comprehensive yet straightforward guide to apply CAM and Grad-CAM for CNN explanation and visualization. Adjust target layers, models, and visualization methods according to your project needs

