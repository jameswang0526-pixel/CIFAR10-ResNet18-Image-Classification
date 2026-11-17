# CIFAR-10 Image Classification with ResNet-18

A modular deep learning project that implements a **ResNet-18 convolutional neural network (CNN) from scratch using PyTorch** to classify images from the CIFAR-10 dataset.

The project covers the complete deep learning workflow, including data loading, model architecture, training, evaluation, error analysis, and single-image prediction.

## Project Overview

This project uses a ResNet-18 architecture adapted for the CIFAR-10 dataset.

Unlike using a pretrained model, the ResNet-18 architecture is implemented from scratch and trained on CIFAR-10.

**Key Features**
- Modular project architecture.
- Custom ResNet-18 implementation using PyTorch.
- CIFAR-10 data loading and preprocessing.
- Training and validation pipeline.
- Model checkpointing and training history.
- Test-set evaluation and confusion matrix.
- Per-class accuracy, precision, recall, and F1-score.
- Misclassified image visualization.
- Single-image prediction with top-3 class probabilities.

## Model Architecture

The project implements ResNet-18 with the following structure:

| Component | Configuration |
|---|---|
| Input | 3 × 32 × 32 RGB image |
| Stem | 3 × 3 Convolution, stride 1 |
| Residual Layers | [2, 2, 2, 2] |
| Layer 1 | 64 channels |
| Layer 2 | 128 channels |
| Layer 3 | 256 channels |
| Layer 4 | 512 channels |
| Pooling | Adaptive Average Pooling |
| Classifier | Fully Connected Layer |
| Output | 10 classes |

The architecture is adapted for 32 × 32 CIFAR-10 images by using a 3 × 3 convolution with stride 1 and removing the initial max-pooling layer.

### Residual Learning

ResNet uses skip connections to allow information to bypass convolutional layers.

The residual block follows:

`Output = ReLU(F(x) + Shortcut(x))`

This helps the network learn residual mappings and makes deeper networks easier to optimize.

## Dataset

This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

CIFAR-10 contains 60,000 color images across 10 classes.

| Dataset Split | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |
| **Total** | **60,000** |

The 50,000 official training images are split into training and validation subsets. The official 10,000-image test set is kept separate for final evaluation.

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Project Structure

```text
CIFAR10-ResNet18-Image-Classification/
│
├── dataload/
│   ├── __init__.py
│   ├── cogif.py
│   ├── dataset.py
│   └── transform.py
│
├── Models/
│   └── Models/
│       ├── __init__.py
│       └── resnet.py
│
├── training/
│   ├── __init__.py
│   ├── config.py
│   ├── engine.py
│   └── trainer.py
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
│
├── requirements.txt
├── train.py
├── evaluate.py
├── visualize.py
└── predict.py
```

## Training

The model was trained for 30 epochs using the following configuration:

| Parameter | Value |
|---|---:|
| Model | ResNet-18 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| Batch Size | 64 |
| Epochs | 30 |
| Learning Rate Scheduler | Cosine Annealing |
| Random Seed | 42 |

### Training Results

| Metric | Result |
|---|---:|
| Training Accuracy | 98.61% |
| Validation Accuracy | 92.36% |
| Training Loss | 0.0484 |
| Validation Loss | 0.2541 |

The best model checkpoint is selected based on validation accuracy.


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jameswang0526-pixel/CIFAR10-ResNet18-Image-Classification.git

cd CIFAR10-ResNet18-Image-Classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the CIFAR-10 dataset

Download the CIFAR-10 Python version from:

https://www.cs.toronto.edu/~kriz/cifar.html

Extract the dataset and configure the local dataset path in `dataload/cogif.py`.

The project uses the local CIFAR-10 Python batch files and does not require automatic dataset downloading.

## Usage

### 1. Train the model

```bash
python train.py
```

This will:
- Load and preprocess the dataset.
- Initialize the ResNet-18 model.
- Train the model.
- Evaluate validation performance after each epoch.
- Save the best and latest checkpoints.

### 2. Evaluate the model

```bash
python evaluate.py
```

This will evaluate the best checkpoint on the official CIFAR-10 test set and report:
- Test Loss
- Test Accuracy
- Confusion Matrix
- Per-class Accuracy
- Precision, Recall, and F1-score
- Most Confused Class Pairs

### 3. Visualize misclassified images

```bash
python visualize.py
```

This displays incorrectly classified test images, together with their actual and predicted labels.

### 4. Predict a single image

Set the image path in `predict.py`:

```python
IMAGE_PATH = "my_cat.jpg"
```

Then run:

```bash
python predict.py
```

The script returns:
- Predicted class
- Prediction confidence
- Top-3 predicted classes and probabilities

## Model Weights

The trained model checkpoint is saved as:

```text
runs/resnet18/best.pt
```

Model checkpoints are excluded from the Git repository to avoid unnecessarily increasing repository size.

To use the trained model without retraining:

1. Download `best.pt` from the project's GitHub Releases.
2. Place it inside `runs/resnet18/`.
3. Run `evaluate.py`, `visualize.py`, or `predict.py`.

**GitHub Release:** [Add your model checkpoint release link here]

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)

## Future Improvements

- Build a web-based image classification interface.
- Experiment with data augmentation strategies.
- Tune hyperparameters to improve generalization.
- Analyze class-specific classification errors.
- Compare ResNet-18 with other CNN architectures.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Built with PyTorch | ResNet-18 | CIFAR-10**
