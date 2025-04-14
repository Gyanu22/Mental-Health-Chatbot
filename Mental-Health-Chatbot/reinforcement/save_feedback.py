import os
import csv
from datetime import datetime
from reinforcement.reward_model import retrain_on_new_feedback

# File path for new feedback storage
FEEDBACK_FILE = 'data/new_feedback.csv'

# Updated headers for reinforcement training
HEADERS = [
    'timestamp', 'user_id', 'Input', 'Response', 'Rating', 'Age', 'Gender',
    'Location', 'Emotion', 'Severity_Level', 'Suggested_Action',
    'Followup_Question', 'Reward'
]

def save_feedback(
    user_id, input_text, response_text, rating,
    age=None, gender=None, location=None,
    emotion=None, severity=None, action=None,
    followup=None, reward=None
):
    """Save user feedback into a CSV file for RLHF training"""

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "Input": input_text,
        "Response": response_text,
        "Rating": rating,
        "Age": age,
        "Gender": gender,
        "Location": location,
        "Emotion": emotion,
        "Severity_Level": severity,
        "Suggested_Action": action,
        "Followup_Question": followup,
        "Reward": reward
    }

    # Check if the CSV exists
    write_header = not os.path.exists(FEEDBACK_FILE)

    # Save to CSV
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(feedback_data)

    print("✅ Feedback saved successfully to new_feedback.csv!")

    # Retrain on updated data
    retrain_on_new_feedback()

    return True
