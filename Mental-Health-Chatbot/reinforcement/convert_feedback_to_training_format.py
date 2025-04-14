import os
import pandas as pd
from datetime import datetime

# --- Paths ---
RAW_FEEDBACK_PATH = "data/new_feedback.csv"
TRAINING_FORMAT_PATH = "data/mental_health_embeddings.csv"

# --- Convert and Save ---
def convert_feedback():
    if not os.path.exists(RAW_FEEDBACK_PATH):
        print("❌ No raw feedback to convert.")
        return

    raw_df = pd.read_csv(RAW_FEEDBACK_PATH)

    # Drop rows with missing essentials
    raw_df = raw_df.dropna(subset=["symptoms", "response", "rating"])

    converted_rows = []
    for _, row in raw_df.iterrows():
        converted = {
            "Input": row.get("symptoms", ""),
            "Response": row.get("response", ""),
            "Sub_Topic": "N/A",
            "Specialization": "N/A",
            "Emotion": row.get("emotion", "neutral"),
            "Severity_Level": row.get("severity", "moderate"),
            "Suggested_Action": row.get("action", "acknowledge"),
            "User_Type": "general",
            "Followup_Question": row.get("followup", "N/A"),
            "Input_Embedding": "N/A",  # Will be computed during embedding
            "Reward": row.get("reward", 0)
        }
        converted_rows.append(converted)

    if not converted_rows:
        print("⚠️ No valid rows to convert.")
        return

    df_converted = pd.DataFrame(converted_rows)

    if os.path.exists(TRAINING_FORMAT_PATH):
        df_existing = pd.read_csv(TRAINING_FORMAT_PATH)
        df_combined = pd.concat([df_existing, df_converted], ignore_index=True)
    else:
        df_combined = df_converted

    df_combined.to_csv(TRAINING_FORMAT_PATH, index=False)
    print(f"✅ Converted feedback saved to {TRAINING_FORMAT_PATH}.")

if __name__ == "__main__":
    convert_feedback()
