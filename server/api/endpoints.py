from contextlib import asynccontextmanager

from bll.agents.knowledge import KnowledgeAgent
from core.config import config
from dal.mongo_db import MongoDB, MongoRetriever
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class MessageRequest(BaseModel):
    message: str


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for the FastAPI app."""
    app.state.knowledge_agent = KnowledgeAgent(
        chatllm=ChatOllama(model="llama3.2"),
        agent_description="You are working with University Of Obuda.",
        retriever=MongoRetriever(config.mongo.COLLECTION_NAME, top_k=5),
    )

    yield
    MongoDB.close()


app = FastAPI(lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods; restrict in production
    allow_headers=["*"],  # Allow all headers; restrict in production
)


@app.post("/knowledge")
async def knowledge(request: MessageRequest):
    response = await app.state.knowledge_agent.ainvoke(request.message)
    return {"response": response}
