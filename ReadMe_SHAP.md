# SHAP Implementation on Images in Keras
This repository provides an implementation of SHAP (SHapley Additive exPlanations) for interpreting predictions made by Keras models specifically for image classification tasks. SHAP values help to explain the contribution of each pixel to the model's predictions, providing insights into how the model makes decisions.

# Table of Contents
1. Introduction
2. Installation
3. Usage
4. Example Usage
5. References

# Introduction
SHAP is a powerful tool for model interpretation based on cooperative game theory. It assigns each feature (or pixel, in the case of images) an importance value for a particular prediction. This implementation focuses on using SHAP with convolutional neural networks (CNNs) built with Keras, allowing users to visualize which parts of an image contribute most to the model's predictions.

# Installation
1. Clone the repository:
git clone https://github.com/your-username/shap-images-keras.git
cd shap-images-keras
2. Install the required dependencies:
pip install -r requirements.txt

# Usage
1. Import necessary libraries:
   import numpy as np
   import shap
   from tensorflow.keras.applications import ResNet50, preprocess_input
   from tensorflow.keras.preprocessing import image

2. Load a pre-trained model:
base_model = ResNet50(include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)
model.summary()

3. Prepare your input data:
X_train, y_train = load_dataset(train_dir, img_size)
X_test,  y_test  = load_dataset(test_dir, img_size)

4. Define a function to preprocess the input images
def f(X):
    tmp = preprocess_input(X.copy())
    return model.predict(tmp)

5. Set up the SHAP explainer: Define a masker that masks out parts of the input image
masker = shap.maskers.Image("inpaint_telea", X.shape)

6. Create an explainer instance
explainer = shap.Explainer(f, masker)

7. Generate SHAP values:
Calculate SHAP values for selected images
shap_values = explainer(X[1:3], max_evals=500, batch_size=50)

6. Visualize the results:
Plot the SHAP values overlaid on the original images
shap.image_plot(shap_values)

# Example:
SHAP can explain overall feature importance and individual prediction explanations. It is used for understanding model behavior, debugging, and compliance.

# References
1. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 4765-4774).
2. SHAP documentation: SHAP Documentation.
3. Example notebooks and further resources can be found in the official SHAP GitHub repository: SHAP GitHub.
