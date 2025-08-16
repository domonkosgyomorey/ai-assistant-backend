from contextlib import asynccontextmanager

from api.common_types import RequestModel
from bll.agents import NIKObudaKnowledgeAgent
from dal.mongo_db import MongoDB
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_client = MongoDB()
    app.state.knowledge_agent = NIKObudaKnowledgeAgent()
    yield
    app.state.db_client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask/stream")
async def ask_question_stream(request: RequestModel):
    """
    Ask a question to the knowledge agent with streaming response.

    Args:
        request: The question request with structured input

    Returns:
        Streaming response with answer chunks
    """

    async def generate_response():
        agent: NIKObudaKnowledgeAgent = app.state.knowledge_agent

        async for chunk in agent.astream(request.question):
            yield chunk

    return StreamingResponse(generate_response())
