

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1️⃣: Load dataset
file_path = r'D:\fraudTrain.csv'
df = pd.read_csv(file_path, parse_dates=['trans_date_trans_time'])

print("✅ Loaded DataFrame:", df.shape)
print(df['is_fraud'].value_counts())

# Step 2️⃣: Encode target
if df['is_fraud'].dtype != 'int':
    df['is_fraud'] = LabelEncoder().fit_transform(df['is_fraud'])

# Step 3️⃣: Extract datetime features
df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day
df['month'] = df['trans_date_trans_time'].dt.month

# Step 4️⃣: Select features & copy
features = ['amt', 'hour', 'day', 'month']
X = df[features].copy()
y = df['is_fraud']

# Step 5️⃣: Ensure float dtype to avoid FutureWarning
X[features] = X[features].astype(float)

# Step 6️⃣: Scale features
scaler = StandardScaler()
X.loc[:, features] = scaler.fit_transform(X[features])

# Step 7️⃣: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Step 8️⃣: Train model (balanced)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 9️⃣: Predictions & evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print(f"🔍 Fraud cases in test set: {y_test.sum()}")
print(f"✔️ Correctly detected frauds: {((y_test==1)&(y_pred==1)).sum()}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 🔟: Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Fraud Detection – Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
