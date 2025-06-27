 TASK-4: Logistic Regression Binary Classifier

Welcome to **Task 4** — a deep dive into the world of machine learning where we’ve built a **binary classification model** using **Logistic Regression** to predict whether a tumor is **malignant (M)** or **benign (B)** using the Breast Cancer Wisconsin dataset.

---
 Objective

To build a **Logistic Regression** model that can classify tumors as malignant or benign based on various cell nucleus features. This task helps understand the power of logistic regression in solving **binary classification** problems.

---

 Dataset Used

- **Source:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Link:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Filename:** `data.csv`
- **Attributes:** 30 numeric features + 1 target column (`diagnosis`)

---

Technologies & Libraries

- `Python`
- `Pandas`
- `Scikit-learn`
- `Matplotlib`
- `Seaborn`

---

 Steps Performed

1. **Data Loading**  
   Read and explore the dataset.

2. **Preprocessing**  
   - Converted `diagnosis` into binary (`M` = 0, `B` = 1)  
   - Dropped unnecessary columns (`id`, `Unnamed: 32`)

3. **Train-Test Split**  
   Divided the data (80% training, 20% testing)

4. **Feature Scaling**  
   Applied `StandardScaler` to normalize the data

5. **Model Training**  
   Trained a `LogisticRegression` model using Scikit-learn

6. **Model Evaluation**  
   - **Confusion Matrix**
   - **Precision**
   - **Recall**
   - **ROC AUC Score**
   - **ROC Curve Plot**

7. **Threshold Tuning (Optional)**  
   Evaluated the model at a custom threshold of 0.4


| Metric                | Score  |
|-----------------------|--------|
| Precision             | ~0.97  |
| Recall                | ~0.99  |
| ROC AUC Score         | ~1.00  |
| Precision (Threshold=0.4) | ~0.97  |
| Recall (Threshold=0.4)    | ~1.00  |



- Confusion Matrix
- ROC Curve

Both graphs are included in this repository as image files:
- `Figure_1.png`
- `roc curve.png`

Make sure you have Python installed, then:

```bash
pip install pandas matplotlib scikit-learn
python task4_logistic_classifier.py
