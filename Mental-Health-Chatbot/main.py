import streamlit as st
st.set_page_config(page_title="🧠 Mental Health Chatbot", layout="wide")

import sqlite3
import uuid
from chatbot.core import generate_response
from chatbot.database import DB_PATH, init_db, load_chat, save_chat, delete_chat
from reinforcement.save_feedback import save_feedback
from reinforcement.reward_model import load_or_train, retrain_on_new_feedback, predict_reward, train_model

# ------------------ Try loading the model safely ------------------
try:
    model = load_or_train()
except ValueError as e:
    st.warning("⚠️ Reward model not trained yet. Chatbot will still work without it.")
    model = None

# ------------------ Configuration ------------------
st.title("🧠 Mental Health Chatbot 🤖")
st.sidebar.header("📂 Chat History")

# ------------------ Train Reward Model Button ------------------
if st.sidebar.button("⚙️ Train Reward Model"):
    try:
        train_model()
        st.success("✅ Model trained successfully!")
    except ValueError as e:
        st.error(f"❌ Training failed: {e}")

# ------------------ Initialize DB ------------------
init_db()

# ------------------ Load/Set Chat ID ------------------
if "chat_id" not in st.session_state:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id FROM chat_history")
    last_chat = cursor.fetchone()
    conn.close()
    st.session_state.chat_id = last_chat[0] if last_chat else f"chat_{uuid.uuid4()}"
    st.session_state.messages = load_chat(st.session_state.chat_id)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "show_feedback_form" not in st.session_state:
    st.session_state.show_feedback_form = False

# ------------------ Sidebar: Manage Chats ------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT chat_id FROM chat_history")
chat_ids = [row[0] for row in cursor.fetchall()]
conn.close()

selected_chat = st.sidebar.selectbox("🗂️ Select a Chat", chat_ids, 
    index=chat_ids.index(st.session_state.chat_id) if st.session_state.chat_id in chat_ids else 0)

if selected_chat != st.session_state.chat_id:
    st.session_state.chat_id = selected_chat
    st.session_state.messages = load_chat(selected_chat)
    st.rerun()

if st.sidebar.button("🆕 New Chat"):
    new_id = f"chat_{uuid.uuid4()}"
    st.session_state.chat_id = new_id
    st.session_state.messages = []
    save_chat(new_id, str([]))
    st.rerun()

if st.sidebar.button("🗑️ Delete Chat"):
    delete_chat(st.session_state.chat_id)
    st.session_state.chat_id = f"chat_{uuid.uuid4()}"
    st.session_state.messages = []
    st.rerun()

# ------------------ Chat Display ------------------
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if i == len(st.session_state.messages) - 1 and message["role"] == "assistant":
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("👍", key=f"thumbs_up_{i}"):
                save_feedback(st.session_state.chat_id, message["content"], 1, None, None, None, None, None, None, None, None)
                st.success("👍 Thanks for the positive feedback!")
        with col2:
            if st.button("👎", key=f"thumbs_down_{i}"):
                save_feedback(st.session_state.chat_id, message["content"], -1, None, None, None, None, None, None, None, None)
                st.warning("👎 Feedback noted. We'll try to improve!")
        with col3:
            if st.button("📝 Give Feedback", key=f"show_feedback_btn_{i}"):
                st.session_state.show_feedback_form = True

# ------------------ User Input ------------------
user_input = st.chat_input("💬 Ask anything", disabled=st.session_state.processing)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.processing = True

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thinking...")

    bot_response = generate_response(user_input)

    if bot_response.strip():
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.session_state.processing = False
        save_chat(st.session_state.chat_id, st.session_state.messages)
        st.rerun()

# ------------------ Feedback Form ------------------
if st.session_state.show_feedback_form:
    st.markdown("---")
    st.subheader("📝 Detailed Feedback Form")

    with st.form("feedback_form"):
        user_id = str(uuid.uuid4())
        last_input = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), "")
        last_response = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"), "")

        rating = st.slider("🌟 Rate the bot's response (1 = Poor, 5 = Excellent)", 1, 5, 3)
        age = st.number_input("🎂 Age", min_value=10, max_value=100, value=25)
        gender = st.selectbox("🚻 Gender", ["Prefer not to say", "Male", "Female", "Other"])
        location = st.text_input("📍 Location")
        emotion = st.selectbox("😌 Emotion", ["Neutral", "Positive", "Negative"])
        severity = st.selectbox("🔥 Severity", ["Low", "Moderate", "High"])
        action = st.selectbox("💡 Suggested Action", ["Talk to a counselor", "Relaxation exercises", "Seek medical advice"])
        followup = st.text_input("🔁 Any follow-up questions?")

        submitted = st.form_submit_button("✅ Submit Feedback")
        if submitted:
            predicted_reward = predict_reward(last_input, last_response, emotion, severity, action)
            save_feedback(user_id, last_input, last_response, rating, age, gender, location, emotion, severity, action, followup, predicted_reward)
            retrain_on_new_feedback()
            st.success("🎉 Feedback saved. Thanks for helping us improve!")
            st.session_state.show_feedback_form = False
