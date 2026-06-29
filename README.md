# 🔢 Handwritten Digit Recognition — Classifier Comparison

A machine learning project that benchmarks four classification algorithms on the **Optical Recognition of Handwritten Digits** dataset (sklearn built-in).

## 📌 Overview

The goal is to compare the performance of multiple classifiers on an 8×8 pixel grayscale image classification task with 10 classes (digits 0–9).

## 📊 Models Compared

| Model | Notes |
|-------|-------|
| Random Forest | `max_depth=128`, `n_estimators=256` |
| SVM | Default RBF kernel |
| ANN (MLP) | `hidden_layer_sizes=256` |
| KNN | `n_neighbors=8` |

## 🗂️ Dataset

- **Source:** `sklearn.datasets.load_digits`
- **Samples:** 1,797 images
- **Features:** 64 (8×8 pixel values, flattened)
- **Classes:** 10 (digits 0–9)
- **Train/Test split:** 70% / 30%
- **Preprocessing:** MinMax normalization to [0, 1]

## 📈 Evaluation Metrics

Each model is evaluated on:
- Training & Test **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)

## 🚀 How to Run

1. Clone the repo and open the notebook:
   ```bash
   git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   jupyter notebook computer_vision.ipynb
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn matplotlib jupyter
   ```

3. Run all cells top to bottom.

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn
- matplotlib
- Jupyter Notebook

## 📁 Structure

```
├── computer_vision.ipynb   # Main notebook
└── README.md
```
