import pandas as pd
import zipfile
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Download dataset directly from UCI (no manual download)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))

# Step 3: Read the text file inside the ZIP
df = pd.read_csv(z.open("SMSSpamCollection"), sep="\t", names=["label", "message"])

# Step 4: Basic exploration
print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# Step 5: Encode labels (ham = 0, spam = 1)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42
)

# Step 7: Convert text → numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 8: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test_tfidf)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix - SMS Spam Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
