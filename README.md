# 🩺 Breast Cancer Detection

Comparison of seven classical machine learning classifiers on the Wisconsin Diagnostic Breast Cancer dataset, implemented with scikit-learn in a Jupyter notebook.

> ⚠️ Educational project only. This is not a medical tool and must not be used for clinical decisions.

## 🗂️ Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`
- **Size:** 569 samples (212 malignant, 357 benign), 30 numeric features computed from digitized images of breast mass cell nuclei
- **Task:** Binary classification (malignant vs. benign). In scikit-learn's encoding, class `0` is malignant and class `1` is benign.

## ⚙️ Preprocessing

- Stratified 80/20 train/test split with `random_state=42` (455 train / 114 test samples)
- Min-Max scaling to [0, 1], fitted on the training data only and applied to the test data

## 🤖 Models

| Model | Main settings |
| ----- | ------------- |
| Gaussian Naive Bayes (GNB) | defaults |
| K-Nearest Neighbors (KNN) | `n_neighbors=8`, `algorithm='kd_tree'`, `leaf_size=28` |
| Decision Tree (DT) | `max_depth=64`, `criterion='gini'`, `random_state=42` |
| Random Forest (RF) | `n_estimators=1000`, `max_depth=32`, `min_samples_split=4`, `random_state=42` |
| SVM | `kernel='poly'` |
| Logistic Regression (LR) | defaults |
| ANN (MLPClassifier) | `hidden_layer_sizes=1024`, `activation='tanh'`, `solver='lbfgs'`, `random_state=42` |

## 📊 Results

### Single train/test split

Accuracy on the train and test sets; precision and recall on the test set for class `1` (benign).

| Model | Train Acc. | Test Acc. | Precision | Recall |
| ----- | ---------- | --------- | --------- | ------ |
| GNB | 0.9385 | 0.9298 | 0.9444 | 0.9444 |
| KNN | 0.9802 | 0.9561 | 0.9718 | 0.9583 |
| DT | 1.0000 | 0.9123 | 0.9559 | 0.9028 |
| RF | 0.9978 | 0.9561 | 0.9589 | 0.9722 |
| SVM | 0.9868 | 0.9737 | 0.9859 | 0.9722 |
| LR | 0.9780 | 0.9561 | 0.9467 | 0.9861 |
| ANN | 1.0000 | 0.9123 | 0.9559 | 0.9028 |

### Stratified 5-fold cross-validation

The scaler is part of the pipeline, so it is fitted on the training folds only. `Recall (malignant)` is the share of malignant tumors that are detected.

| Model | CV Accuracy (mean ± std) | CV Recall, malignant (mean ± std) |
| ----- | ------------------------ | --------------------------------- |
| GNB | 0.9297 ± 0.0199 | 0.8916 ± 0.0484 |
| KNN | 0.9684 ± 0.0162 | 0.9482 ± 0.0377 |
| DT | 0.9104 ± 0.0279 | 0.8588 ± 0.0737 |
| RF | 0.9561 ± 0.0135 | 0.9295 ± 0.0509 |
| SVM | 0.9772 ± 0.0142 | 0.9529 ± 0.0492 |
| LR | 0.9649 ± 0.0200 | 0.9153 ± 0.0618 |
| ANN | 0.9526 ± 0.0204 | 0.9389 ± 0.0351 |

### Notes and limitations

- With 114 test samples, one sample changes accuracy by about 0.9%. A single split is a noisy basis for ranking models, so the cross-validation table is the more reliable comparison.
- SVM and KNN have the highest cross-validated accuracy and malignant recall, but the gaps between the top models are within one standard deviation.
- Decision Tree and ANN reach 100% training accuracy with lower test accuracy, which suggests overfitting.
- Hyperparameters were set by hand and were not tuned.

## 🚀 How to Run

```
git clone https://github.com/amirh-ganji/breast-cancer-detection.git
cd breast-cancer-detection
pip install scikit-learn matplotlib jupyter
jupyter notebook breast_cancer_detection.ipynb
```

Run all cells from top to bottom. Results are seeded, but small differences can appear across scikit-learn versions.

## 📁 Structure

```
├── breast_cancer_detection.ipynb   # Data loading, preprocessing, models, comparison, cross-validation
├── LICENSE
└── README.md
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
