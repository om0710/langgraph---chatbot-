from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage

# Persistence
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # Take query from user
    messages = state["messages"]

    # Send to LLM
    response = llm.invoke(messages)

    # Store response in state
    return {
        "messages": [response]
    }

conn = sqlite3.connect(database='chatbot.db' , check_same_thread=False)
memory = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile with persistence
workflow = graph.compile(checkpointer=memory)
thread_id = "1"
config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
workflow.get_state(config)
def retrieve_all_threads():
    all_threads = set()

    for checkpoint in memory.list(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)