import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import os

from backend_rag import workflow, retrieve_all_threads
from rag import add_pdf_to_vectordb

# ---------------- Query Params Sync ----------------
params = st.query_params
if "thread_id" in params:
    selected_thread = params["thread_id"]
    st.session_state.thread_id = selected_thread
    st.session_state.current_chat = selected_thread

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Intelligent Chat Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Streamlit UI Custom CSS Injection ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Set font only on text elements, avoiding Streamlit SVG icons & material symbols */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    h1, h2, h3, h4, p, textarea, input {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    span:not([class*="icon"]):not([class*="Icon"]):not([class*="material"]):not([class*="symbol"]),
    button:not([class*="icon"]):not([class*="Icon"]):not([class*="material"]) {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Global Background Gradient */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 20%, rgb(18, 16, 32) 0%, rgb(7, 5, 14) 90%) !important;
        color: #f8fafc !important;
    }
    
    /* Hide Streamlit default decoration, deploy bar, and options menu */
    [data-testid="stDeployButton"], .stDeployButton, [data-testid="stMainMenu"], #MainMenu, footer {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    
    /* Make header background transparent */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Optimize padding to prevent top header from clipping the main title */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0 !important;
    }
    
    /* Main container max width limit for chat interface */
    [data-testid="stBlock"] {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 8, 22, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding: 24px 16px !important;
    }
    
    /* Logo Styling */
    .logo {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 24px;
        padding-left: 8px;
    }
    
    /* Custom HTML Thread Items styling matching HTML5 mockup exactly */
    .thread-item {
        display: flex;
        align-items: center;
        gap: 12px;
        background-color: rgba(255, 255, 255, 0.02);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 500;
        font-size: 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .thread-item:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: #f8fafc;
        border-color: rgba(255, 255, 255, 0.12);
    }
    .thread-item.active {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(79, 70, 229, 0.1)) !important;
        border-color: rgba(124, 58, 237, 0.4) !important;
        color: #c084fc !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.05) !important;
    }
    
    /* Primary Button Custom Styling (+ New Chat) */
    div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.2) !important;
        justify-content: center !important;
        text-align: center !important;
        margin-bottom: 12px !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
        color: white !important;
    }
    
    /* Sidebar Headers style */
    [data-testid="stSidebar"] h3 {
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
        letter-spacing: 0.05em !important;
        margin: 0 0 12px 0 !important;
        padding-left: 8px !important;
        border: none !important;
    }
    
    /* Sidebar horizontal lines */
    [data-testid="stSidebar"] hr {
        margin: 12px 0 !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Chat Message Bubbles styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-bottom-right-radius: 4px !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.15) !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: #f8fafc !important;
        border-bottom-left-radius: 4px !important;
    }
    
    /* Code blocks customization */
    code {
        color: #f472b6 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    pre {
        background-color: rgba(15, 12, 30, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
    }
    
    /* Knowledge Base PDF Uploader Box styling */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.01) !important;
        border: 2.5px dashed rgba(124, 58, 237, 0.35) !important;
        border-radius: 12px !important;
        padding: 28px 16px !important;
        transition: all 0.3s ease !important;
        margin-top: 12px !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #c084fc !important;
        background-color: rgba(124, 58, 237, 0.04) !important;
    }
    
    /* Hide Streamlit default dropzone file uploader labels/buttons */
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] section > input + div {
        display: none !important;
    }
    [data-testid="stFileUploader"] section > div {
        display: none !important;
    }
    
    /* Inject custom cloud upload icon */
    [data-testid="stFileUploader"] section::before {
        content: "" !important;
        display: block !important;
        width: 36px !important;
        height: 36px !important;
        margin: 0 auto 10px auto !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-size: contain !important;
        opacity: 0.8;
    }
    
    /* Inject custom upload instruction text */
    [data-testid="stFileUploader"] section::after {
        content: "Drag & Drop PDF or Click to Upload" !important;
        color: #94a3b8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        display: block !important;
        text-align: center !important;
        margin-top: 10px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Chat Input Bar style */
    [data-testid="stChatInput"] {
        background-color: rgba(15, 12, 30, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 30px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(15px) !important;
        padding: 8px 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
    }
    
    /* Style Chat Input Send Button to be purple and rounded */
    [data-testid="stChatInput"] button {
        background-color: #7c3aed !important;
        color: white !important;
        border-radius: 50% !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #8b5cf6 !important;
        transform: scale(1.05) !important;
    }
    
    /* Restore icon font rendering for collapse arrow button */
    [data-testid="collapsedControl"] * {
        font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State Initialization ----------------
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "thread_id" not in st.session_state:
    if st.session_state.chat_threads:
        st.session_state.thread_id = st.session_state.chat_threads[0]
    else:
        st.session_state.thread_id = str(uuid.uuid4())

if st.session_state.thread_id not in st.session_state.chat_threads:
    st.session_state.chat_threads.insert(0, st.session_state.thread_id)

if "current_chat" not in st.session_state:
    st.session_state.current_chat = st.session_state.thread_id

if "messages" not in st.session_state:
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }
    state = workflow.get_state(config)
    if state.values and "messages" in state.values:
        st.session_state.messages = state.values["messages"]
    else:
        st.session_state.messages = []

# ---------------- Sidebar Layout ----------------
with st.sidebar:
    st.markdown(
        """
        <div class="logo" style="font-family: 'Plus Jakarta Sans', sans-serif;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-message-square-code"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="m10 8-2 2 2 2"/><path d="m14 8 2 2-2 2"/></svg>
            <span style="color: #ffffff;">Assistant </span><span style="color: #c084fc;">Portal</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_thread = str(uuid.uuid4())
        st.session_state.thread_id = new_thread
        st.session_state.current_chat = new_thread
        st.session_state.messages = []
        if new_thread not in st.session_state.chat_threads:
            st.session_state.chat_threads.append(new_thread)
        st.rerun()

    st.divider()
    st.markdown("### 💬 ACTIVE SESSIONS")

    # List chat threads inside a scrollable container
    with st.container(height=320):
        for thread in st.session_state.chat_threads:
            short_id = thread[:8] + "..." if len(thread) > 8 else thread
            is_current = (thread == st.session_state.thread_id)
            
            active_class = "active" if is_current else ""
            
            # HTML Link-based thread switcher with exact visual styles matching client mockup
            st.markdown(
                f"""
                <a href="?thread_id={thread}" target="_self" style="text-decoration: none;">
                    <div class="thread-item {active_class}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                        <span>Chat {short_id}</span>
                    </div>
                </a>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ---------------- Upload PDF (Knowledge Base) in Sidebar ----------------
    st.markdown("### 📄 KNOWLEDGE BASE")
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
        st.success("✅ Document context uploaded and indexed!")

# ---------------- Main Chat Area ----------------
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 16px;">
        <div style="font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; font-family: 'Plus Jakarta Sans', sans-serif;">Intelligent Chat <span style="color: #c084fc;">Agent</span></div>
        <div style="background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 30px; padding: 6px 14px; font-size: 12px; color: #94a3b8; font-weight: 500; font-family: 'Plus Jakarta Sans', sans-serif;">Session: {st.session_state.thread_id[:8]}...</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Chat History ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)

# ---------------- Chat Input ----------------
user_input = st.chat_input("Ask anything...")

if user_input:
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant", avatar="🤖"):
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