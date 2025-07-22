

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load data from CSV into DataFrame
df = pd.read_csv(r"D:\Churn_Modelling.csv")
print("✅ Data loaded:", df.shape)
print("Columns available:", df.columns.tolist())

# 2. Drop rows with missing values (if any)
df = df.dropna()

# 3. Pick the correct column names
# In this dataset, column names are 'Tenure', 'Balance', 'EstimatedSalary', and 'Exited'
# Let's use 'Tenure' (years with the bank) and 'EstimatedSalary' (salary), and target is 'Exited'
X = df[['Tenure', 'EstimatedSalary']].astype(float)
y = df['Exited']

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=1
)

# 5. Standardize features
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 6. Train model using Logistic Regression
model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(X_train_s, y_train)

# 7. Make predictions and print the results
y_pred = model.predict(X_test_s)

print("\n=== Logistic Regression Results ===")
print(classification_report(y_test, y_pred, zero_division=0))
