from contextlib import asynccontextmanager

from api.common_types import RequestModel
from bll.agents.knowledge import Knowledge
from dal.mongo_db import MongoDB
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_client = MongoDB()
    app.state.knowledge_agent = Knowledge()
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


@app.post("/prompt")
async def prompt(request: RequestModel):
    messages = [request.message] + request.context
    messages = [msg.to_langchain_message() for msg in messages]

    agent: Knowledge = app.state.knowledge_agent

    context = await agent.ainvoke({"messages": messages})
    answer = context["answer"]
    response_chunk = {"message": {"role": "ai", "content": answer}, "metadata": context}

    return response_chunk


@app.post("/health")
async def health_check():
    return {"status": "healthy"}
