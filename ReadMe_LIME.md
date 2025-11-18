# LIME Implementation in Keras
This repository provides an implementation of the LIME (Local Interpretable Model-agnostic Explanations) algorithm to explain predictions made by Keras models. LIME is a popular technique for interpreting complex machine learning models, including deep neural networks, by approximating their behavior locally with simpler models.

# Table of Contents
1. Introduction
2. Installation
3. Usage
     Example Usage
4. References

# Introduction
LIME helps in understanding the predictions of machine learning models by generating locally faithful explanations. It works by perturbing the input data and observing the changes in predictions, thereby allowing us to understand which features are most influential. This implementation is suitable for both image and text classification tasks using Keras.

# Installation
1. Clone the repository:
git clone https://github.com/your-username/lime-keras.git
cd lime-keras

2. Install the required dependencies:
pip install -r requirements.txt

# Usage
1. Import the necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from lime import lime_image
from lime.lime_text import LimeTextExplainer

2. Load your pre-trained Keras model:
model = load_model('path/to/your/model.h5')

3. For image classification, create a LIME image explainer:
explainer = lime_image.LimeImageExplainer()

4. For text classification, create a LIME text explainer:
text_explainer = LimeTextExplainer(class_names=['class1', 'class2', ...])

5. Generate explanations for an image:
def predict_fn(images):
    return model.predict(images)

explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)

6. Generate explanations for text:
explanation = text_explainer.explain_instance(text_instance, model.predict, num_features=10)

7. Visualize the results for image classification:
from skimage.segmentation import mark_boundaries

temp, mask = explainer.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
img_boundry = mark_boundaries(temp / 255.0, mask)

plt.imshow(img_boundry)
plt.axis('off')
plt.show()

8. Visualize the results for text classification:
word_idx = explanation.as_list()

# Example Usage
You can find an example usage in the LIME.ipynb file provided in this repository.

# References
1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Link to paper.
2. LIME GitHub Repository: LIME.