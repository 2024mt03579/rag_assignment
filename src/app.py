import streamlit as st
from rag import create_vector_store
from qa import answer_with_rag
from utils import detect_intent, call_llm

# Optional: If you want to use the intent model instead of LLM-based detection
#from intent_model import predict_intent
#from utils import call_llm

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="AI Healthcare Assistant")

# Header
st.title("🩺 AI Healthcare Assistant")
st.warning("⚠️ For educational use only. Not medical/legal advice.")

# Load Vector DB
@st.cache_resource
def load_db():
    return create_vector_store()

db = load_db()

# Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # ✅ Show sources attached to this message
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown("### 📄 Sources")
            for s in set(msg["sources"]):
                st.caption(f"📄 {s}")

# User Input
if prompt := st.chat_input("Ask anything (healthcare, quiz, compliance, summary)..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Detect intent
    intent = detect_intent(prompt)

    # Optional: If using a custom intent model instead of LLM-based detection
    #intent = predict_intent(prompt)

    logging.info(f"Detected intent: {intent}")

    # Default → RAG
    response, sources = answer_with_rag(db, prompt)

    # Override for special tasks
    if intent == "SUMMARIZE":
        response = call_llm(f"Summarize:\n{prompt}")

    elif intent == "QUIZ":
        response = call_llm(f"Generate 3 quiz questions:\n{prompt}")

    elif intent == "COMPLIANCE":
        response = call_llm(f"""
        Check HIPAA and GDPR compliance:

        {prompt}

        Provide:
        - Compliance status
        - Risks
        - Suggestions
        """)

    with st.chat_message("assistant"):
        st.markdown(response)

    # Show sources immediately
    if sources:
        st.markdown("### 📄 Sources")
        for s in set(sources):
            st.caption(f"📄 {s}")

    # Save message WITH sources
    st.session_state.messages.append({
    "role": "assistant",
    "content": response,
    "sources": sources
    })