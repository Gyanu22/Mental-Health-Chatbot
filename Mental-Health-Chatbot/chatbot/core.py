import os
import numpy as np
import pandas as pd
import sqlite3
import hashlib
import streamlit as st
import google.generativeai as genai

from chatbot.embeddings import load_resources
from chatbot.database import DB_PATH

# Load FAISS index, DataFrame, and embedding model
index, df, embedding_model = load_resources()

# Configure Gemini
genai.configure(api_key="AIzaSyDsqpgL_hClxO8FDFO3QmaclGbq7__AJ0w")  # Replace with secure method for deployment
llm = genai.GenerativeModel("gemini-2.0-flash")

# ------------------ FAISS Search ------------------

def retrieve_from_faiss(query, top_k=3):
    """Retrieve most relevant responses from FAISS index."""
    query_embedding = np.array([embedding_model.encode(query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]]["Response"].dropna().tolist() or ["I'm not sure how to respond to that."]

# ------------------ Response Caching ------------------

@st.cache_data
def get_cached_response(user_input):
    """Check if an identical query has already been answered."""
    input_hash = hashlib.md5(user_input.encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM chat_history WHERE chat_id = ?", (st.session_state.chat_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        messages = eval(row[0])
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                return msg["content"]
    return None

# ------------------ Chatbot Response Generation ------------------

def generate_response(user_input):
    """Generate a natural, context-aware response using FAISS + Gemini."""
    
    # Retrieve previous conversation history (last 5 turns)
    chat_history = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in st.session_state.messages[-5:]
    ])

    # Semantic search from FAISS
    retrieved = retrieve_from_faiss(user_input)
    retrieved_text = "\n".join([f"- {r}" for r in retrieved])

    # Gemini Prompt
    prompt = f"""
    You are a compassionate AI assistant trained to support mental health.
    Use the following context to generate a meaningful response:

    --- Chat History ---
    {chat_history}

    --- User's Latest Message ---
    {user_input}

    --- Related Past Knowledge ---
    {retrieved_text}

    Respond in a calm, empathetic, and professional tone. Do not mention the history explicitly.
    """

    # Generate and return response
    try:
        output = llm.generate_content(prompt)
        return output.text.strip()
    except Exception as e:
        return "I'm having trouble thinking clearly right now. Could you please try again?"
