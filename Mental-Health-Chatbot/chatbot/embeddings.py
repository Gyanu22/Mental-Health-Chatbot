import os
import pandas as pd
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import streamlit as st

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)

# ------------------ Paths & Globals ------------------
CSV_PATH = "data/mental_health_embeddings.csv"
FEEDBACK_PATH = "data/user_feedback.csv"
INDEX_PATH = "assets/mental_health_faiss.index"
MODEL_NAME = "all-MiniLM-L6-v2"

index = None
texts = []

# ------------------ Model Loader ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

embedding_model = load_embedding_model()

# ------------------ FAISS Build Function ------------------
def build_faiss_index():
    global index, texts

    base_df = pd.read_csv(CSV_PATH)

    if os.path.exists(FEEDBACK_PATH):
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        feedback_df = feedback_df.rename(columns={"symptoms": "Input"})
        base_df = pd.concat([base_df, feedback_df[["Input"]]])

    texts = base_df["Input"].dropna().tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    logging.info("✅ FAISS index built and saved.")

# ------------------ FAISS Load Function ------------------
def load_faiss_index():
    global index, texts

    df = pd.read_csv(CSV_PATH)

    if os.path.exists(FEEDBACK_PATH):
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        feedback_df = feedback_df.rename(columns={"symptoms": "Input"})
        df = pd.concat([df, feedback_df[["Input"]]])

    texts = df["Input"].dropna().tolist()
    index = faiss.read_index(INDEX_PATH)
    logging.info("📦 FAISS index loaded from disk.")

# ------------------ Startup: Build or Load ------------------
if not os.path.exists(INDEX_PATH):
    build_faiss_index()
else:
    try:
        load_faiss_index()
    except Exception as e:
        logging.warning(f"Failed to load FAISS index: {e}, rebuilding...")
        build_faiss_index()

# ------------------ FAISS Query ------------------
def query_faiss(query, top_k=3):
    vec = embedding_model.encode([query])
    D, I = index.search(np.array(vec).astype("float32"), top_k)
    return [texts[i] for i in I[0]]

# ------------------ Streamlit-Compatible Loader ------------------
@st.cache_resource
def load_resources():
    local_index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(CSV_PATH)
    df["Input_Embedding"] = df["Input_Embedding"].apply(eval) if "Input_Embedding" in df.columns else None
    return local_index, df, embedding_model
