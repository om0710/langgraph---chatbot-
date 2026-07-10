import os
import json
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import LangGraph workflow and config
from backend_rag import workflow, retrieve_all_threads
from rag import add_pdf_to_vectordb
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="LangGraph Chatbot Client API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str

def serialize_message(msg):
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        # Only return AIMessages that have content (skipping tool-call triggers)
        if msg.content:
            return {"role": "assistant", "content": msg.content}
    return None

# Serve static frontend folder
os.makedirs("static", exist_ok=True)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/threads")
def get_threads():
    try:
        threads = retrieve_all_threads()
        return {"threads": threads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thread/{thread_id}/history")
def get_thread_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = workflow.get_state(config)
        messages = []
        if state.values and "messages" in state.values:
            for msg in state.values["messages"]:
                serialized = serialize_message(msg)
                if serialized:
                    messages.append(serialized)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    state = {"messages": [HumanMessage(content=request.query)]}
    
    async def response_generator():
        last_msg_id = None
        last_len = 0
        try:
            # Run the langgraph workflow streaming
            for chunk in workflow.stream(state, config=config, stream_mode="values"):
                ai_message = chunk["messages"][-1]
                if isinstance(ai_message, AIMessage):
                    # Reset character tracking if we transitioned to a new AIMessage in the list
                    msg_obj_id = id(ai_message)
                    if msg_obj_id != last_msg_id:
                        last_msg_id = msg_obj_id
                        last_len = 0
                        
                    content = ai_message.content
                    if len(content) > last_len:
                        new_chunk = content[last_len:]
                        last_len = len(content)
                        yield f"data: {json.dumps({'text': new_chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to vector DB with clean text parsing
        add_pdf_to_vectordb(file_path)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
