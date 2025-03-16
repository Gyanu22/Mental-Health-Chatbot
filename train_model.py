import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_FILE = "final_refined_mental_health_final.csv"
MODEL_FILE = "classification_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Load dataset
df = pd.read_csv(DATASET_FILE)
X = df["Input"]
y = df["Response"]

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model and vectorizer
joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)

print("✅ Model training complete!")
