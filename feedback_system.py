import pandas as pd

DATASET_FILE = "final_refined_mental_health_final.csv"

def collect_feedback(user_input, chatbot_response):
    """Stores user feedback directly in the main dataset if response needs improvement."""
    feedback = input("Was this response helpful? (yes/no): ").strip().lower()
    
    if feedback == "no":
        print("❌ Storing feedback for improvement...")
        
        df = pd.read_csv(DATASET_FILE)

        if user_input in df["Input"].values:
            df.loc[df["Input"] == user_input, "Response"] = "Needs Improvement"
        else:
            new_entry = pd.DataFrame({"Input": [user_input], "Response": ["Needs Improvement"]})
            df = pd.concat([df, new_entry], ignore_index=True)

        df.to_csv(DATASET_FILE, index=False)
        print("✅ Feedback stored in dataset!")

    else:
        print("✅ Positive feedback received!")

if __name__ == "__main__":
    test_input = "I feel very lonely."
    test_response = "Try talking to a friend or engaging in an activity you enjoy."
    collect_feedback(test_input, test_response)
