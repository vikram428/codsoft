import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = r"D:\train_data.txt"
data = []
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            _, _, genre, plot = parts
            data.append((plot.strip(), genre.strip().lower()))

# Create DataFrame
df = pd.DataFrame(data, columns=["plot", "genre"])

# Filter genres with enough samples
min_count = 100
genre_counts = df['genre'].value_counts()
df = df[df['genre'].isin(genre_counts[genre_counts >= min_count].index)]

# Print samples and class distribution
print("Sample Rows:")
print(df.head())
print("\nGenre Distribution:")
print(df['genre'].value_counts())

# Feature and label
X = df['plot']
y = df['genre']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix - Green Style 🌿
plt.figure(figsize=(14, 10))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='Greens',  # ✅ Changed here
            xticklabels=sorted(model.classes_),
            yticklabels=sorted(model.classes_))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.tight_layout()
plt.show()
