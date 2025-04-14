import os
import joblib
from reinforcement.reward_model import retrain_on_new_feedback, calculate_reward

model_path = "reinforcement/reward_model.pkl"

# Check if the model exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Loaded existing model.")
else:
    print("⚠️ Warning: reward_model.pkl not found. Training a new model.")
    model = None

# Retrain the model if necessary
retrain_on_new_feedback(model, calculate_reward)
