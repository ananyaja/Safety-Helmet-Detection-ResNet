# Safety-Helmet-Detection-ResNet
Safety-Helmet-Detection-ResNet is an end-to-end computer vision pipeline designed to automate industrial safety compliance by accurately classifying the presence of protective safety helmets in images.
# Safety-Helmet-Detection-ResNet

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision pipeline developed to automate industrial safety compliance by accurately classifying the presence of protective safety helmets in images. Built using **PyTorch**, this project leverages a fine-tuned **ResNet18** deep convolutional neural network backbone to extract complex spatial patterns and optimize object boundary detection.

---

## 🚀 Key Engineering Highlights

* **Strategic Fine-Tuning:** Unfroze the deep convolutional layers (`layer4`) of a pre-trained ResNet18 network to allow localized weight adaptation for specialized safety gear features, successfully resolving an initial categorical baseline prediction collapse.
* **Data Augmentation & Regularization:** Implemented a robust data manipulation pipeline (including random rotations, horizontal flips, and color jitter) to prevent the network from exploiting background macro-features, ensuring highly reliable real-world model generalization.
* **Production Quality Controls:** Integrated a custom PyTorch dataset wrapper featuring dynamic dimension shape auto-detection along with an early stopping optimization loop based on cross-validation loss tracing to completely eliminate overfitting.
* **Performance Metrics:** Reached stable, high-confidence classification boundaries on out-of-sample data splits, successfully exporting verified inference predictions for Kaggle evaluation.

---

## 🛠️ Tech Stack & Dependencies

* **Core Language:** Python
* **Deep Learning Framework:** PyTorch, Torchvision
* **Data Science & Analytics:** NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib

---

## 📈 Model Performance & Evaluation

The pipeline utilizes an automatic early stopping checkpoint that saves the model parameters at the absolute lowest point of validation loss. This prevents semantic decay and ensures the model is frozen at the peak of its generalization ability.

### Final Validation Report
```text
Confusion Matrix:
[[64  0]
 [ 0 63]]

Classification Metrics:
              precision    recall  f1-score   support

   No Helmet       1.00      1.00      1.00        64
      Helmet       1.00      1.00      1.00        63

    accuracy                           1.00       127
   macro avg       1.00      1.00      1.00       127
weighted avg       1.00      1.00      1.00       127
