import os
import json
import uuid
import streamlit as st
from pathlib import Path
from rag_generate import rag_answer

# -----------------------------
# Config & paths
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

DATA_DIR = "data/pdfs"
STORAGE_DIR = "storage"
CHATS_FILE = f"{STORAGE_DIR}/chats.json"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers: persistence
# -----------------------------
def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chats():
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, indent=2)

# -----------------------------
# Session state
# -----------------------------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# -----------------------------
# SIDEBAR: PDF Upload
# -----------------------------
with st.sidebar:
    st.header("PDFs")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload"
    )

    active_pdfs = []

    if uploaded:
        for f in uploaded:
            path = os.path.join(DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            active_pdfs.append(f.name)

        st.success(f"{len(uploaded)} PDF(s) uploaded")

# -----------------------------
# SIDEBAR: Chat Controls
# -----------------------------
with st.sidebar:
    st.divider()
    st.header("Chats")

    if st.button("New Chat"):
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = {
            "title": "New Chat",
            "pdfs": active_pdfs,
            "messages": []
        }
        st.session_state.active_chat = chat_id
        save_chats()
        st.rerun()

    for cid, chat in st.session_state.chats.items():
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat["title"], key=f"open_{cid}"):
                st.session_state.active_chat = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = None
                save_chats()
                st.rerun()

# -----------------------------
# MAIN: Active chat
# -----------------------------
if not st.session_state.active_chat:
    st.info("Start a new chat to begin")
    st.stop()

chat = st.session_state.chats[st.session_state.active_chat]

# Rename chat
new_title = st.text_input(
    "Chat title",
    value=chat["title"],
    key="rename_chat"
)
if new_title != chat["title"]:
    chat["title"] = new_title
    save_chats()

# Show PDFs
if chat["pdfs"]:
    st.caption("PDFs: " + ", ".join(chat["pdfs"]))

# Render messages
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat input (MUST BE LAST)
# -----------------------------
query = st.chat_input("Ask a question...", key="chat_input")

if query:
    chat["messages"].append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream, sources = rag_answer(
            query,
            chat["messages"]
        )

        for token in stream:
            if '"response":' in token:
                text = token.split('"response":"')[-1].rstrip('"}')
                full_response += text
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    chat["messages"].append(
        {"role": "assistant", "content": full_response}
    )

    # Auto-title chat if first question
    if chat["title"] == "New Chat":
        chat["title"] = query[:40] + "..."

    save_chats()
    st.rerun()
