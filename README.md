# 🩺 Breast Cancer Detection

Comparison of seven classical machine learning classifiers on the Wisconsin Diagnostic Breast Cancer dataset, implemented with scikit-learn in a Jupyter notebook.

> ⚠️ Educational project only. This is not a medical tool and must not be used for clinical decisions.

## 🗂️ Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic), loaded via `sklearn.datasets.load_breast_cancer`
- **Size:** 569 samples, 30 numeric features computed from digitized images of breast mass cell nuclei
- **Task:** Binary classification (malignant vs. benign)

## ⚙️ Preprocessing

- 80/20 train/test split (455 train / 114 test samples)
- Min-Max scaling to [0, 1], fitted on the training set only and applied to the test set

## 🤖 Models

| Model | Main settings |
| ----- | ------------- |
| Gaussian Naive Bayes (GNB) | defaults |
| K-Nearest Neighbors (KNN) | `n_neighbors=8`, `algorithm='kd_tree'`, `leaf_size=28` |
| Decision Tree (DT) | `max_depth=64`, `criterion='gini'` |
| Random Forest (RF) | `n_estimators=1000`, `max_depth=32`, `min_samples_split=4` |
| SVM | `kernel='poly'` |
| Logistic Regression (LR) | defaults |
| ANN (MLPClassifier) | `hidden_layer_sizes=1024`, `activation='tanh'`, `solver='lbfgs'` |

## 📊 Results

Accuracy on train and test sets; precision and recall on the test set for class `1` (benign).

| Model | Train Acc. | Test Acc. | Precision | Recall |
| ----- | ---------- | --------- | --------- | ------ |
| GNB | 0.9297 | 0.9561 | 0.9851 | 0.9429 |
| KNN | 0.9714 | 0.9912 | 1.0000 | 0.9857 |
| DT | 1.0000 | 0.9386 | 0.9846 | 0.9143 |
| RF | 0.9978 | 0.9912 | 1.0000 | 0.9857 |
| SVM | 0.9824 | 0.9825 | 0.9857 | 0.9857 |
| LR | 0.9714 | 0.9737 | 0.9718 | 0.9857 |
| ANN | 1.0000 | 0.9561 | 0.9851 | 0.9429 |

**Notes and limitations**

- Results come from a single train/test split with 114 test samples, so a difference of one sample changes accuracy by about 0.9%. Differences between the top models are within that noise.
- Decision Tree and ANN reach 100% training accuracy with lower test accuracy, which suggests overfitting.

## 🚀 How to Run

```
git clone https://github.com/amirh-ganji/breast-cancer-detection.git
cd breast-cancer-detection
pip install scikit-learn matplotlib jupyter
jupyter notebook breast_cancer_detection.ipynb
```

Run all cells from top to bottom.

## 📁 Structure

```
├── breast_cancer_detection.ipynb   # Data loading, preprocessing, models, comparison plots
└── README.md
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
