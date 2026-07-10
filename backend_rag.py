from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver

import sqlite3

# Import retriever from your RAG file
from rag import retriever
# Import new tools
from tools import calculator, wikipedia_search, get_current_time, get_stock_price

load_dotenv()

# ---------------- LLM ---------------- #

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def grade_document(question: str, document_content: str) -> str:
    prompt = f"""System: You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the user question, reply with 'yes', otherwise reply with 'no'.
Do not write any other explanation. Just write 'yes' or 'no'.

Question: {question}
Document: {document_content}

Relevance:"""
    try:
        response = llm.invoke(prompt)
        score = response.content.strip().lower()
        if "yes" in score:
            return "yes"
        return "no"
    except Exception:
        return "yes"

def is_general_knowledge_query(question: str) -> bool:
    prompt = f"""System: You are an assistant classifying whether a user question is a general knowledge question (which can be answered by Wikipedia/general web search, e.g., science, history, geography, famous people, general definitions) OR a query specific to local uploaded files, resumes, certificates, or personal context (which CANNOT be answered by Wikipedia).

Reply with 'general' if it is a general knowledge question.
Reply with 'local' if it is about a specific uploaded file, certificate, resume, personal data, or local context.

Do not write any other explanation. Just write 'general' or 'local'.

Question: {question}

Classification:"""
    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower()
        if "general" in classification:
            return True
        return False
    except Exception:
        return False

@tool
def rag_tool(query: str):
    """
        Retrieve relevant information from the PDF document.
    Use this tool when the user asks factual or conceptual
    questions that might be answered from the stored documents.

    """

    docs = retriever.invoke(query)
    
    # Deduplicate retrieved documents to save tokens and prevent context repetition
    seen_contents = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
    docs = unique_docs

    is_general = is_general_knowledge_query(query)
    
    relevant_docs = []
    need_search = False

    if is_general:
        if not docs:
            need_search = True
        else:
            for doc in docs:
                score = grade_document(query, doc.page_content)
                if score == "yes":
                    relevant_docs.append(doc)

            if not relevant_docs:
                need_search = True
    else:
        relevant_docs = docs
        need_search = (len(docs) == 0)

    context_list = [doc.page_content for doc in relevant_docs]
    
    if need_search:
        # Check if it is a general knowledge query before triggering Wikipedia fallback
        if is_general:
            try:
                search_result = wikipedia_search.invoke(query)
                context_list.append(f"[Corrective Search Fallback Context] {search_result}")
            except Exception:
                pass
        else:
            # For local/document-specific questions, do not fall back to Wikipedia search
            context_list.append("[No relevant information found in the uploaded documents. Do not attempt to guess or use web search for local document questions.]")

    context = "\n\n".join(context_list)
    return context


tools = [rag_tool, calculator, wikipedia_search, get_current_time, get_stock_price]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

# ---------------- State ---------------- #

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------- Chat Node ---------------- #

def prune_messages(messages: list, max_count: int = 8) -> list:
    # 1. Clean up past ToolMessages to remove massive context blocks
    cleaned_messages = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            # If it is not the very last message in the history, replace its content with a placeholder
            if i < len(messages) - 1:
                name = getattr(msg, "name", "tool")
                msg = ToolMessage(
                    content=f"[Context retrieved from {name}]",
                    name=name,
                    tool_call_id=msg.tool_call_id
                )
        cleaned_messages.append(msg)
    
    messages = cleaned_messages

    if len(messages) <= max_count:
        return messages
    
    sliced = messages[-max_count:]
    while sliced and isinstance(sliced[0], ToolMessage):
        idx = len(messages) - len(sliced)
        if idx > 0:
            sliced.insert(0, messages[idx - 1])
        else:
            break
    return sliced

def chat_node(state: ChatState):

    system_instruction = SystemMessage(
        content=(
            "You are a helpful and friendly assistant. System instructions:\n"
            "- Do NOT call any tools (especially `wikipedia_search` or `rag_tool`) for personal introductions, greetings, chit-chat, or questions about the user's name (e.g. 'my name is om', 'hii', 'what is my name?'). Just reply directly as a friendly chat.\n"
            "- For questions requiring PDF contents or general knowledge, you MUST first call the appropriate tool (`rag_tool` or `wikipedia_search`) to retrieve the context. Only say that you do not know if the tools return no information.\n"
            "- For stock prices/quotes/values, you MUST call the `get_stock_price` tool. The returned value metric represents its stock price in USD. You MUST explicitly output this value metric to the user as the actual stock price (e.g., if the tool returns 'Value metric for AAPL: 316.22', your response MUST state that the stock price is $316.22). Do NOT hide, omit, or replace it with a disclaimer, and do NOT claim the value is simulated or fictional. Report it as the actual stock price.\n"
            "- Only call `rag_tool` for queries about uploaded files (resumes, certificates, documents).\n"
            "- The candidate/student's name on a certificate is the person who 'successfully completed' the course (e.g., 'Om Bansal'). Do NOT confuse the student with directors, signers, or authorizers listed at the bottom (like 'Amanda Brophy').\n"
            "- Call `calculator` for math, `wikipedia_search` for general knowledge, and `get_current_time` for system time.\n"
            "- IMPORTANT: You MUST include the exact numbers, names, and facts returned by tools in your response. Do not censor or alter them."
        )
    )

    pruned = prune_messages(state["messages"])
    messages = [system_instruction] + pruned

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ---------------- SQLite ---------------- #

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

memory = SqliteSaver(conn)

# ---------------- Graph ---------------- #

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_node("tools", tool_node)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)

workflow = graph.compile(
    checkpointer=memory
)

# ---------------- Thread Utility ---------------- #

def retrieve_all_threads():

    threads = []

    for checkpoint in memory.list(None):

        thread = checkpoint.config["configurable"]["thread_id"]

        if thread not in threads:
            threads.append(thread)

    return threads