# Breast Cancer Classification – ML Models + Streamlit App

## a. Problem Statement
Predict whether a breast tumour is **Malignant (M)** or **Benign (B)** using measurements computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. This is a **binary classification** problem. Five classification models are trained on the same dataset, compared on six evaluation metrics, and demonstrated through an interactive Streamlit web app.

## b. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository (also on Kaggle)
- **Instances:** 569
- **Features:** 30 numeric features (+ an `id` column, dropped before training)
- **Target:** `diagnosis` – M = Malignant (212), B = Benign (357). Encoded as M = 1, B = 0.
- **Features:** 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as mean (`_mean`), standard error (`_se`) and worst value (`_worst`), giving 30 features.
- **Missing values:** None
- **Split:** 80% train / 20% test (stratified, `random_state=7`). The 114-row test split is saved as `test_data.csv`.
- **Preprocessing:** StandardScaler fitted on training data only, saved as `model/scaler.pkl`.

## c. GitHub Repository Link
- GitHub: `https://github.com/mukund-bits/ML-Assignment2`
- Streamlit App: `<add app link here>`

## d. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (k = 5)
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (RBF kernel)

### Comparison Table (on test data, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9954 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Decision Tree | 0.9211 | 0.9077 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| KNN | 0.9474 | 0.9808 | 0.9737 | 0.8810 | 0.9250 | 0.8872 |
| Naive Bayes | 0.9123 | 0.9752 | 0.9211 | 0.8333 | 0.8750 | 0.8101 |
| SVM | 0.9649 | 0.9954 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |

### Overall Winner
**Logistic Regression** – it has the highest Accuracy, Precision, F1 and MCC, and ties with SVM for the best AUC and Recall.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall. Precision of 1.0 means no benign tumour was wrongly called malignant. The scaled features are close to linearly separable, so a simple linear model works very well. |
| Decision Tree | Weakest AUC (0.908) because a single tree gives hard, near 0/1 probabilities. It also has lower recall; an unpruned single tree tends to overfit the training data. |
| KNN | Good results after scaling (accuracy 0.947), but recall (0.881) is lower than LR and SVM, so it misses more malignant cases. |
| Naive Bayes | Lowest accuracy, recall and MCC. Many features are strongly correlated (e.g. radius, perimeter, area), which breaks the independence assumption. Its AUC is still high, so its ranking of cases is decent. |
| SVM | Very close second. Same AUC and recall as LR, with slightly lower precision. The RBF kernel handles the data well but gives no real gain over the linear model here. |
