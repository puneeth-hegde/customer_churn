import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# 1️⃣ Load Dataset
df = pd.read_csv("data/churn.csv")
print("Data Shape:", df.shape)
print(df.head())

# 2️⃣ Basic EDA
print("\nMissing Values:\n", df.isnull().sum())
print("\nChurn Distribution:\n", df['Churn'].value_counts())

# Save churn distribution plot
os.makedirs("outputs", exist_ok=True)
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.close()

# 3️⃣ Preprocessing
df = df.drop(['customerID'], axis=1)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5️⃣ Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Evaluate Model
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# Save feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.close()

# Save classification report
with open("outputs/churn_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# 7️⃣ Save Model + Preprocessing Artifacts
artifacts = {
    "model": model,
    "scaler": scaler,
    "feature_names": X.columns
}
pickle.dump(artifacts, open("churn_model.pkl", "wb"))
print("\n✅ Training complete! Model & artifacts saved as 'churn_model.pkl'.")
