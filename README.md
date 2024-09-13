# Black-box Attack

This repository contains an implementation of a black-box adversarial attack using the approach described in the paper "[*Transferable Adversarial Attack based on Integrated Gradients*](https://arxiv.org/abs/2205.13152)" and visualizes clean and adversarial images using `captum` for explainability.

## Files

- **`blackbox_attack.py`**: Implements the blackbox adversarial attack where:
  - **Surrogate Model**: ResNet50.
  - **Victim Model**: ResNet18 trained on the RIVAL10 dataset. 
    - Reference for RIVAL10: "[*A Comprehensive Study of Image Classification Model Sensitivity to Foregrounds, Backgrounds, and Visual Attributes*](https://arxiv.org/abs/2201.10766)".
  - The victim model achieved an accuracy of **93%** on clean images.
  - The adversarial attack achieved an attack success rate (ASR) of **8%**.
  - **Note**: 1000 samples from the RIVAL10 dataset were selected for the blackbox attack.

- **`captum.ipynb`**: Jupyter Notebook demonstrating the visualization of a sample clean image and its adversarial counterpart using two visualization techniques from the `captum` library:
  - **Guided GradCAM**.
  - **Guided Backpropagation**.

## Dataset

The victim model was trained on the **RIVAL10** dataset, which is a challenging image classification dataset designed for testing sensitivity to image fore- and background, as well as visual attributes. The surrogate model (ResNet50) was used to generate adversarial examples targeting the ResNet18 victim model.

## Dependencies

This project requires the following libraries:
- PyTorch
- Captum (for model explainability)
- NumPy
- Matplotlib

To install the required libraries, you can use the following command:

```bash
pip install torch captum numpy matplotlib 
