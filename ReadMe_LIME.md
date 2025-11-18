# LIME Implementation in Keras
This repository provides an implementation and usage examples of LIME, a popular technique to explain predictions of any machine learning classifier or regressor by learning an interpretable model locally around the prediction.

# Table of Contents
1. Introduction
2. Installation
3. Usage
4. Example Usage
5. References

# Introduction
LIME explains individual predictions by creating synthetic data samples around the instance and fitting a simple, interpretable model (e.g., linear) weighted by the proximity to the original instance. This reveals which features most influence the model's prediction locally.

LIME is model-agnostic, supports tabular, text, and image data, and helps identify why complex models make certain decisions.

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
   
explainer = lime_image.LimeImageExplainer(kernel_width=0.15, random_state=42)
explanation = explainer.explain_instance(
    img, 
    model.predict, 
    labels=(0, ),
    hide_color=None, 
    top_labels=4, 
    num_features=1000, 
    num_samples=1000, 
    batch_size=10,
    segmentation_fn=None, 
    distance_metric='cosine', 
    model_regressor=None, 
    random_seed=10)

5. Generate explanations for an image:
def predict_fn(images):
    return model.predict(images)

explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)

6. Visualize the results for image classification:
from skimage.segmentation import mark_boundaries

temp, mask = explainer.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
img_boundry = mark_boundaries(temp / 255.0, mask)

plt.imshow(img_boundry)
plt.axis('off')
plt.show()

# Example Usage
You can find an example usage in the SHAP_Result.ipynb file provided in this repository.

# References
1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Link to paper.

2. LIME GitHub: https://github.com/marcotcr/lime.
