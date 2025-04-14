import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

# --- Paths ---
CSV_PATH = "data/mental_health_embeddings.csv"
VECTOR_PATH = "reinforcement/vectorizer.pkl"
MODEL_PATH = "reinforcement/reward_model.pkl"
ENCODER_PATH = "reinforcement/encoder.pkl"
FEEDBACK_PATH = "data/new_feedback.csv"

# --- Reward Calculation Logic ---
def calculate_reward(row):
    reward = 0
    reward += 1 if row["Severity_Level"] in ["moderate", "high"] else -1
    if row["Emotion"] == "positive":
        reward += 1
    elif row["Emotion"] == "negative":
        reward -= 1
    if "great" in row["Response"].lower():
        reward += 1
    elif "bad" in row["Response"].lower():
        reward -= 1
    return 1 if reward >= 1 else 0

# --- Load and Prepare Training Data ---
def load_data(path):
    """Load and prepare data for training."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {path} not found!")

    df = pd.read_csv(path)

    # Drop rows with missing essential columns
    df = df.dropna(subset=["Input", "Response", "Emotion", "Severity_Level", "Suggested_Action"])

    # Create combined text feature for training
    df["combined_text"] = df["Input"] + " " + df["Response"]
    df["reward"] = df["Response"].apply(assign_reward)  # Apply the new robust reward assignment

    # Check the distribution of feedback
    check_feedback_distribution(df)

    return df

def check_feedback_distribution(df):
    """Print the distribution of reward classes."""
    reward_counts = df["reward"].value_counts()
    print("Feedback Distribution:")
    print(f"Positive feedback (1): {reward_counts.get(1, 0)}")
    print(f"Negative feedback (0): {reward_counts.get(0, 0)}")
    return reward_counts

# --- Vectorize and Encode ---
def extract_features(df, vectorizer=None, encoder=None, fit=False):
    text = df["combined_text"]

    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(text)
        joblib.dump(vectorizer, VECTOR_PATH)
    text_features = vectorizer.transform(text)

    cats = df[["Emotion", "Suggested_Action", "Severity_Level"]].fillna("unknown")

    if fit or encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(cats)
        joblib.dump(encoder, ENCODER_PATH)
    cat_features = encoder.transform(cats)

    X = np.hstack([text_features.toarray(), cat_features])
    y = df["reward"]

    return X, y, vectorizer, encoder

# --- Training ---
def train_model():
    df = load_data(CSV_PATH)
    X, y, vectorizer, encoder = extract_features(df, fit=True)

    if len(np.unique(y)) < 2:
        raise ValueError("Need samples of at least 2 reward classes to train.")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("✅ Reward model trained and saved.")
    return model, vectorizer, encoder

# --- Load or Train ---
def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTOR_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Loaded existing reward model.")
        return model
    else:
        print("⚠️ No reward model found. Training a new one.")
        model, _, _ = train_model()
        return model

# --- Retrain on New Feedback ---
def retrain_on_new_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        print("ℹ️ No new feedback found.")
        return

    try:
        feedback_df = load_data(FEEDBACK_PATH)
        if feedback_df.empty:
            print("⚠️ Feedback CSV is empty.")
            return

        vectorizer = joblib.load(VECTOR_PATH)
        encoder = joblib.load(ENCODER_PATH)
        X_new, y_new, _, _ = extract_features(feedback_df, vectorizer, encoder)

        if len(np.unique(y_new)) < 2:
            print("⚠️ Not enough reward class variety to retrain.")
            return

        model = joblib.load(MODEL_PATH)
        model.fit(X_new, y_new)
        joblib.dump(model, MODEL_PATH)
        print("✅ Retrained the model with new feedback.")

    except Exception as e:
        print(f"❌ Error during retraining: {e}")

# --- Prediction Helper ---
def predict_reward(input_text, response_text, emotion, severity, action):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTOR_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except (FileNotFoundError, NotFittedError):
        print("⚠️ Model not found. Please train it first.")
        return 0

    combined = input_text + " " + response_text
    X_text = vectorizer.transform([combined])
    X_cat = encoder.transform([[emotion, action, severity]])
    X = np.hstack([X_text.toarray(), X_cat])

    return model.predict(X)[0]

# --- Assign Reward Logic ---
def assign_reward(feedback_text):
    """Assign a reward based on feedback text."""
    feedback_text = feedback_text.lower()
    if "good" in feedback_text or "positive" in feedback_text or "excellent" in feedback_text:
        return 1  # Positive feedback
    elif "bad" in feedback_text or "negative" in feedback_text or "poor" in feedback_text:
        return 0  # Negative feedback
    else:
        return 0  # Default to 0 for neutral feedback 
