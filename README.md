## README.md Template

# ML Classification Streamlit App

## Problem Statement
Predict whether a patient has breast cancer (Malignant or Benign) using multiple ML classification algorithms.

## Dataset Description
The dataset is the built-in sklearn Breast Cancer dataset with 569 instances and 30 numeric features. The target is binary: 0 for Malignant and 1 for Benign.

## Models Used and Metrics
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|------|------|------|------|------|------|
| Logistic Regression | 0.982456 | 0.995370 | 0.982456 | 0.982456 | 0.982456 | 0.962302 |
| Decision Tree | 0.912281 | 0.915675 | 0.916072 | 0.912281 | 0.913021 | 0.817412 |
| KNN | 0.956140 | 0.978836 | 0.956073 | 0.956140 | 0.956027 | 0.905447 |
| Naive Bayes | 0.929825 | 0.986772 | 0.929825 | 0.929825 | 0.929825 | 0.849206 |
| Random Forest | 0.947368 | 0.994709 | 0.947368 | 0.947368 | 0.947368 | 0.886905 |
| XGBoost | 0.947368 | 0.991733 | 0.947440 | 0.947368 | 0.947087 | 0.886414 |

## Observations on Model Performance
| ML Model Name | Observation about model performance |
|---------------|-----------------------------------|
| Logistic Regression | Very high accuracy, handles linear separation well, fast to train |
| Decision Tree | Good for capturing nonlinear patterns, slightly overfits on training data |
| KNN | High accuracy, sensitive to feature scaling, performs well with standardized data |
| Naive Bayes | Fast and robust, assumes feature independence, slightly lower accuracy |
| Random Forest | Ensemble method, stable and accurate, reduces overfitting compared to single tree |
| XGBoost | Ensemble boosting, slightly lower than Random Forest in this dataset, strong generalization |

