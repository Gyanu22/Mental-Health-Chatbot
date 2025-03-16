
import streamlit as st
import pandas as pd
import faiss
import numpy as np
import joblib
import sqlite3
from sentence_transformers import SentenceTransformer
from sentiment_analysis import detect_sentiment
from feedback_system import collect_feedback
from reinforcement_learning import retrain_model
from retrieval_system_updated import retrieve_response

# ------------------ UI Configuration ------------------
st.set_page_config(page_title="Mental Health Chatbot", layout="wide")

st.title("🧠 Mental Health Chatbot 🤖")
st.sidebar.header("📂 Chat History")

# ------------------ Database Setup ------------------
DB_PATH = "chat_database.db"

def init_db():
    """Initialize SQLite database for chat storage"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_chat(chat_id, messages):
    """Save chat history to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO chat_history (chat_id, messages) VALUES (?, ?)", (chat_id, str(messages)))
    conn.commit()
    conn.close()

def load_chat(chat_id):
    """Retrieve chat history from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM chat_history WHERE chat_id = ?", (chat_id,))
    row = cursor.fetchone()
    conn.close()
    return eval(row[0]) if row else []

def delete_chat(chat_id):
    """Delete a chat conversation from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

# ------------------ Sidebar - Chat History Management ------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT chat_id FROM chat_history")
chat_ids = [row[0] for row in cursor.fetchall()]
conn.close()

if not chat_ids:
    chat_ids.append("chat_1")

selected_chat = st.sidebar.selectbox("Select a Chat", chat_ids, index=0)

if selected_chat:
    st.session_state.chat_id = selected_chat
    st.session_state.messages = load_chat(selected_chat)

if st.sidebar.button("🆕 New Chat"):
    new_chat_id = f"chat_{len(chat_ids) + 1}"
    chat_ids.append(new_chat_id)
    st.session_state.chat_id = new_chat_id
    st.session_state.messages = []
    save_chat(new_chat_id, str([]))
    st.rerun()

if st.sidebar.button("🗑️ Delete Chat"):
    delete_chat(st.session_state.chat_id)
    chat_ids.remove(st.session_state.chat_id)
    st.session_state.chat_id = chat_ids[0] if chat_ids else "chat_1"
    st.session_state.messages = load_chat(st.session_state.chat_id)
    st.rerun()

# ------------------ Chat UI ------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_chat(st.session_state.chat_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
user_input = st.chat_input("Ask Anything")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("💬 Thinking...")

    sentiment = detect_sentiment(user_input)
    bot_response = retrieve_response(user_input)

    if sentiment == "Negative":
        bot_response += " (I'm here for you, please take care 💙)"

    if bot_response.strip():
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        message_placeholder.markdown(bot_response)

    # Feedback System
    feedback = st.radio("Was this response helpful?", ("Yes", "No"))

    if feedback == "No":
        new_response = retrieve_response(user_input)  # Regenerate response
        collect_feedback(user_input, new_response)
        st.write("Your feedback has been recorded! 🙏")

    # Regenerate Response Button
    if st.button("🔄 Regenerate Response"):
        new_response = retrieve_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": new_response})
        st.write(new_response)

    save_chat(st.session_state.chat_id, st.session_state.messages)

# ------------------ Admin Section ------------------
st.sidebar.subheader("🔄 Admin Options")
if st.sidebar.button("♻️ Retrain Model with Feedback"):
    retrain_model()
    st.sidebar.success("Model retrained successfully!")
