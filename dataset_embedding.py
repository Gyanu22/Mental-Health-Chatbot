# from sentence_transformers import SentenceTransformer
# import pandas as pd

# # Load the dataset
# file_path = "final_refined_mental_health_dataset.csv"
# df = pd.read_csv(file_path)

# # Remove 'Therapist_Name' column
# df = df.drop(columns=["Therapist_Name"])

# # Load the SentenceTransformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Generate embeddings for "Input" column
# df["Input_Embedding"] = df["Input"].apply(lambda text: embedding_model.encode(text).tolist())

# # Save the processed dataset with embeddings
# df.to_csv("mental_health_embeddings.csv", index=False)

# print("Embedding generation completed successfully!")

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = "final_refined_mental_health_final.csv"
df = pd.read_csv(file_path)

# Drop 'Therapist_Name' as requested
df = df.drop(columns=["Therapist_Name"], errors="ignore")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
df["Input_Embedding"] = df["Input"].apply(lambda text: embedding_model.encode(text).tolist())

# Convert embeddings to numpy array
embeddings = np.array(df["Input_Embedding"].tolist()).astype('float32')

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # 384 dimensions

# Add embeddings to FAISS index
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "mental_health_faiss.index")

# Save processed dataset with embeddings
df.to_csv("mental_health_embeddings.csv", index=False)

print("✅ Data processing complete! Embeddings stored in FAISS.")
