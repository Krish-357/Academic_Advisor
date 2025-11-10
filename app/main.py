import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agents import AgentManager
from typing import Optional

app = FastAPI(title="Personalized Academic & Career Guidance - Agentic AI")

# Allow frontend (Vercel) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryPayload(BaseModel):
    user_id: str
    message: str
    context: Optional[dict] = None

agent_manager = AgentManager()

async def handle_query(payload: QueryPayload):
    try:
        context = payload.context or {}
        result = await agent_manager.handle_user_message(payload.user_id, payload.message, context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Original /api/query
@app.post("/api/query")
async def query(payload: QueryPayload):
    return await handle_query(payload)

# Alias /api/chat
@app.post("/api/chat")
async def chat_alias(payload: QueryPayload):
    return await handle_query(payload)

@app.get("/test")
async def test():
    return {"ok": True}
