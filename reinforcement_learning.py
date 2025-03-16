import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

DATASET_FILE = "final_refined_mental_health_final.csv"
MODEL_FILE = "classification_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

def retrain_model():
    """Retrains the classification model using updated dataset (includes user feedback)."""
    try:
        print("🔄 Retraining model with new feedback...")

        df = pd.read_csv(DATASET_FILE)

        df = df[df["Response"] != "Needs Improvement"]

        if df.empty:
            print("⚠️ No valid data to retrain model. Skipping...")
            return

        X = df["Input"]
        y = df["Response"]

        try:
            vectorizer = joblib.load(VECTORIZER_FILE)
        except:
            vectorizer = TfidfVectorizer()
        
        X_tfidf = vectorizer.fit_transform(X)

        try:
            model = joblib.load(MODEL_FILE)
        except:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_tfidf, y)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)

        print("✅ Model successfully retrained with new feedback data!")

    except Exception as e:
        print(f"⚠️ Error during retraining: {e}")

if __name__ == "__main__":
    retrain_model()
