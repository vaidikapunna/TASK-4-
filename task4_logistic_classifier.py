import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, roc_auc_score, roc_curve
)
print(" Loading the data...")
df = pd.read_csv('data.csv')
print(f" The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n")
print("🧹 Cleaning up and preparing the data...")
if 'diagnosis' in df.columns:
    df['target'] = df['diagnosis'].map({'M': 0, 'B': 1})
    df.drop(['diagnosis'], axis=1, inplace=True)
df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore', inplace=True)
if 'target' not in df.columns:
    raise ValueError(" No 'target' column found! Make sure it's created properly.")
print(" Data is ready for training.\n")
X = df.drop('target', axis=1)
y = df['target']

print(" Splitting into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(" Scaling the features so everyone gets a fair shot...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n Training the Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(" Training complete!\n")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print(" Evaluating performance...\n")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(" Confusion Matrix")
plt.show()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f" Precision: {precision:.2f}")
print(f" Recall:    {recall:.2f}")
print(f" ROC AUC:   {roc_auc:.2f}\n")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC Curve', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(" ROC Curve")
plt.legend()
plt.show()
custom_threshold = 0.4
y_custom = (y_prob >= custom_threshold).astype(int)
print(f"\n Custom Threshold set to {custom_threshold}")
print(f" Precision (Custom): {precision_score(y_test, y_custom):.2f}")
print(f" Recall (Custom):    {recall_score(y_test, y_custom):.2f}")
print("\n Logistic Regression done ")
