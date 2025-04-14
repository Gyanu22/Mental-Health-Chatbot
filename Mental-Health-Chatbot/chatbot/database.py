import sqlite3
import os
import uuid
import json
import pandas as pd
from datetime import datetime
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# ------------------ Paths ------------------
DB_PATH = "data/chat_database.db"
FEEDBACK_PATH = "data/user_feedback.csv"
MODEL_PATH = "reinforcement/reward_model.pkl"
VECTORIZER_PATH = "reinforcement/vectorizer.pkl"

# ------------------ DB Initialization ------------------
def init_db():
    if not os.path.exists("data"):
        os.makedirs("data")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            user_input TEXT,
            ai_response TEXT,
            feedback TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------ Chat Session Management ------------------
def create_new_chat():
    chat_id = str(uuid.uuid4())
    messages = []
    save_chat(chat_id, messages)
    return chat_id

def save_chat(chat_id, messages):
    try:
        formatted_messages = json.dumps(messages, ensure_ascii=False)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("REPLACE INTO chat_history (chat_id, messages) VALUES (?, ?)",
                       (chat_id, formatted_messages))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat: {e}")

def load_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM chat_history WHERE chat_id = ?", (chat_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        try:
            messages = json.loads(row[0])
            return messages if isinstance(messages, list) else []
        except json.JSONDecodeError:
            print("Error decoding JSON. Resetting chat history.")
            return []
    return []

def delete_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

# ------------------ Feedback Storage ------------------
def store_feedback(chat_id, user_input, ai_response, feedback_text):
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (id, chat_id, user_input, ai_response, feedback, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (feedback_id, chat_id, user_input, ai_response, feedback_text, timestamp))
    conn.commit()
    conn.close()

    # Save to CSV for retraining and FAISS
    new_feedback = pd.DataFrame([{
        "symptoms": user_input.strip(),
        "response": ai_response.strip(),
        "feedback": feedback_text.strip(),
        "timestamp": timestamp
    }])

    if not os.path.exists(FEEDBACK_PATH):
        new_feedback.to_csv(FEEDBACK_PATH, index=False)
    else:
        existing = pd.read_csv(FEEDBACK_PATH)
        combined = pd.concat([existing, new_feedback], ignore_index=True)
        combined.to_csv(FEEDBACK_PATH, index=False)

    # Predict the reward after storing feedback
    reward = predict_reward(user_input, ai_response, feedback_text)
    print(f"Predicted Reward: {reward}")

# ------------------ Model and Vectorizer ------------------
def load_or_train_model():
    # Load feedback data
    feedback_data = pd.read_csv(FEEDBACK_PATH) if os.path.exists(FEEDBACK_PATH) else pd.DataFrame()

    if feedback_data.empty:
        print("⚠️ No feedback data found for training.")
        return None

    # Prepare the feedback data
    feedback_data["combined_text"] = feedback_data["symptoms"] + " " + feedback_data["response"]
    feedback_data["reward"] = feedback_data["feedback"].apply(lambda x: 1 if "positive" in x.lower() else 0)

    # Train vectorizer
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Loaded existing vectorizer.")
    else:
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(feedback_data["combined_text"])
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("✅ Trained and saved new vectorizer.")

    # Transform text features
    text_features = vectorizer.transform(feedback_data["combined_text"])

    # Encode categorical features (feedback)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_cats = encoder.fit_transform(feedback_data[["feedback"]])

    # Final feature set
    X = np.hstack([text_features.toarray(), encoded_cats])
    y = feedback_data["reward"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print("✅ Trained and saved new reward model.")
    return model

def predict_reward(input_text, response_text, feedback_text):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    combined = input_text + " " + response_text
    X_text = vectorizer.transform([combined])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform([[feedback_text]])

    final_X = np.hstack([X_text.toarray(), X_cat])

    return model.predict(final_X)[0]

# ------------------ Retraining with New Feedback ------------------
def retrain_on_new_feedback():
    feedback_data = pd.read_csv(FEEDBACK_PATH) if os.path.exists(FEEDBACK_PATH) else pd.DataFrame()

    if len(feedback_data) > 0:
        feedback_data["combined_text"] = feedback_data["symptoms"] + " " + feedback_data["response"]
        feedback_data["reward"] = feedback_data["feedback"].apply(lambda x: 1 if "positive" in x.lower() else 0)

        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(feedback_data["combined_text"])

        text_features = vectorizer.transform(feedback_data["combined_text"])

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_cats = encoder.fit_transform(feedback_data[["feedback"]])

        X = np.hstack([text_features.toarray(), encoded_cats])
        y = feedback_data["reward"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("✅ Retrained the model with new feedback.")
    else:
        print("⚠️ No new feedback to retrain.")

# Example usage: retrain when new feedback is available
retrain_on_new_feedback()
