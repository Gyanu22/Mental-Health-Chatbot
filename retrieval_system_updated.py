
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load FAISS index and dataset with embeddings
dataset_path = "final_refined_mental_health_final.csv"
df = pd.read_csv(dataset_path)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index (if available)
faiss_index_path = "mental_health_faiss.index"
try:
    index = faiss.read_index(faiss_index_path)
except:
    index = None

# Function to retrieve the best response
def retrieve_response(user_query):
    """Retrieves the best response from FAISS or generates a general AI-based response."""
    query_embedding = np.array([embedding_model.encode(user_query)]).astype('float32')
    
    if index:
        distances, indices = index.search(query_embedding, 1)
        best_match = df.iloc[indices[0][0]]["Response"]
        
        if distances[0][0] < 0.5:
            return best_match

    general_responses = [
        "That's an interesting thought. Tell me more about it!",
        "I get what you're saying. How do you feel about it?",
        "That makes sense. What do you think the next step should be?",
        "I see your point. How can I help you with this?",
        "That’s a tough one. What are your thoughts on it?"
    ]
    
    return np.random.choice(general_responses)
