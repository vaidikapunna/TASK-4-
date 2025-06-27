Task 4: Logistic Regression - Binary Classifier

This project builds a **binary classification model** using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset**.

---

Objective

To classify tumors as **Malignant (M)** or **Benign (B)** using logistic regression, and to evaluate the model using standard metrics.

Tools Used

- 🐼 Pandas
- 🎨 Matplotlib
- 🤖 Scikit-learn (LogisticRegression, metrics)
- 🧪 Jupyter / Python

---

Files

- `task4_logistic_classifier.py` – Full Python code
- `data.csv` – Breast Cancer dataset
- `README.md` – Project description

---

 Steps Performed

1. **Loaded** the dataset and inspected its structure
2. **Preprocessed** data: converted labels, dropped unused columns
3. **Split** data into training and testing sets (80/20)
4. **Standardized** features using `StandardScaler`
5. **Trained** a logistic regression model
6. **Evaluated** using:
   - Confusion Matrix
   - Precision, Recall, ROC-AUC Score
   - ROC Curve Plot
7. **Tuned** the decision threshold and discussed the **sigmoid function**

Results

| Metric              | Value |
|---------------------|-------|
| Precision           | 0.97  |
| Recall              | 0.99  |
| ROC AUC             | 1.00  |
| Precision (0.4 threshold) | 0.97 |
| Recall (0.4 threshold)    | 1.00 |

 Key Learnings

- Logistic Regression is ideal for binary classification.
- The **sigmoid function** maps values to a 0–1 probability scale.
- Evaluation metrics like **precision**, **recall**, and **ROC-AUC** are essential.
- **Threshold tuning** can adjust sensitivity vs. specificity.


