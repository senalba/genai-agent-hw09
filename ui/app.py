import streamlit as st
import requests
import os
import uuid

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="GenAI PDF Agent", layout="wide")
st.title("📄 GenAI PDF Agent. HW09. by Vasyl A.")
st.markdown("""
This agent can answer questions about a PDF document you upload. 
It uses a custom tool to count the pages and a retriever to find relevant information.
""")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.agent_ready = False

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    indexing_mode = st.radio(
        "Indexing Mode",
        ("Replace existing context", "Add to existing context"),
        captions=("Start a new session with this PDF only.", "Add this PDF to the current session's knowledge.")
    )

    if st.button("Index PDF"):
        if uploaded_file is not None:
            with st.spinner("Processing and indexing PDF... This may take a moment."):
                mode = "add" if "Add" in indexing_mode else "replace"
                files = {"pdf_file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"mode": mode}
                try:
                    response = requests.post(f"{API_URL}/index_pdf", files=files, data=data, timeout=600)
                    if response.status_code == 200:
                        st.session_state.agent_ready = True
                        if mode == "replace":
                            st.session_state.messages = [] # Clear history only when replacing
                        st.success(f"✅ PDF indexed in '{mode}' mode! The agent is ready.")
                    else:
                        st.error(f"🚨 Indexing failed: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"🚨 Connection error: Could not reach the backend at {API_URL}. Is it running? Error: {e}")
        else:
            st.warning("⚠️ Please upload a PDF file first.")

st.header("2. Chat with the Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the PDF..."):
    if not st.session_state.agent_ready:
        st.warning("⚠️ Please upload and index a PDF before asking questions.", icon="👆")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {"question": prompt, "session_id": st.session_state.session_id}
                response = requests.post(f"{API_URL}/query", json=payload)
                answer = response.json().get("answer", "Sorry, I couldn't get an answer.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})