import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os

from backend_rag import workflow, retrieve_all_threads
from rag import add_pdf_to_vectordb

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Custom CSS Injection ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: radial-gradient(circle at 10% 20%, rgb(18, 16, 32) 0%, rgb(7, 5, 14) 90%) !important;
        color: #f1f5f9 !important;
    }
    
    /* App Header styling */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
    }
    
    /* Main container max width limit for chat interface */
    [data-testid="stBlock"] {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(12, 10, 24, 0.8) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding: 20px 10px;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 40%, #c084fc 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 2rem !important;
        letter-spacing: -0.025em !important;
        text-align: center !important;
    }
    
    /* Buttons Custom Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.15) !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Sidebar Thread buttons: Default and Active styling */
    .sidebar-btn-active button {
        background: linear-gradient(135deg, #c084fc, #7c3aed) !important;
        box-shadow: 0 4px 20px rgba(192, 132, 252, 0.25) !important;
    }
    
    /* Chat Message Bubbles styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 18px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    .stChatMessage:hover {
        border-color: rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 6px 35px rgba(0, 0, 0, 0.2) !important;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(79, 70, 229, 0.08)) !important;
        border: 1px solid rgba(124, 58, 237, 0.25) !important;
        box-shadow: 0 4px 30px rgba(124, 58, 237, 0.05) !important;
    }
    
    /* Avatar Icons background removal */
    [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Code blocks theme customization */
    code {
        color: #f472b6 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    pre {
        background-color: rgba(15, 12, 30, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
    }
    
    /* File Uploader Container styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1.5px dashed rgba(124, 58, 237, 0.35) !important;
        border-radius: 14px !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #c084fc !important;
        background-color: rgba(124, 58, 237, 0.04) !important;
    }
    
    /* Input box wrapper */
    [data-testid="stChatInput"] {
        background-color: rgba(10, 8, 22, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 28px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(20px) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State Init ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

# Add current thread if it doesn't already exist
if st.session_state.thread_id not in st.session_state.chat_threads:
    st.session_state.chat_threads.append(st.session_state.thread_id)

if "current_chat" not in st.session_state:
    st.session_state.current_chat = st.session_state.thread_id

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 💬 Threads")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_thread = str(uuid.uuid4())
        st.session_state.thread_id = new_thread
        st.session_state.current_chat = new_thread
        st.session_state.messages = []
        if new_thread not in st.session_state.chat_threads:
            st.session_state.chat_threads.append(new_thread)
        st.rerun()

    st.divider()

    # List chat threads
    for thread in st.session_state.chat_threads:
        short_id = thread[:8] + "..." if len(thread) > 8 else thread
        is_current = (thread == st.session_state.thread_id)
        
        # Dynamic label prefix to represent status
        label = f"✨ Active: {short_id}" if is_current else f"💬 Chat {short_id}"
        
        # Style active thread differently
        if is_current:
            st.markdown('<div class="sidebar-btn-active">', unsafe_allow_html=True)
            
        if st.button(label, key=f"btn_{thread}", use_container_width=True):
            st.session_state.thread_id = thread
            st.session_state.current_chat = thread
            
            config = {
                "configurable": {
                    "thread_id": thread
                }
            }
            state = workflow.get_state(config)
            if state.values:
                st.session_state.messages = state.values["messages"]
            else:
                st.session_state.messages = []
            st.rerun()
            
        if is_current:
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ---------------- Upload PDF (Knowledge Base) in Sidebar ----------------
    st.markdown("### 📄 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload PDF context",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Analyzing PDF..."):
            add_pdf_to_vectordb(pdf_path)
        st.success("✅ PDF context added!")

# ---------------- Main Chat Area ----------------
st.markdown('<h1 class="main-title">🤖 Assistant Agent</h1>', unsafe_allow_html=True)

# ---------------- Chat History ----------------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------------- Chat Input ----------------
user_input = st.chat_input("Type your message...")

if user_input:
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in workflow.stream(
            {
                "messages": [user_msg]
            },
            config=config,
            stream_mode="values"
        ):
            ai_message = chunk["messages"][-1]
            if isinstance(ai_message, AIMessage):
                full_response = ai_message.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append(
        AIMessage(content=full_response)
    )

    # Refresh sidebar thread list from SQLite
    st.session_state.chat_threads = retrieve_all_threads()