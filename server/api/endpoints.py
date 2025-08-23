from contextlib import asynccontextmanager

from api.common_types import RequestModel
from bll.agents.knowledge import Knowledge
from dal.mongo_db import MongoDB
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


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


@app.post("/ask/stream")
async def ask_question_stream(request: RequestModel):
    async def generate_response():
        agent: Knowledge = app.state.knowledge_agent

        langchain_messages = [msg.to_langchain_message() for msg in request.messages]

        async for chunk in agent.astream({"messages": langchain_messages}):
            if isinstance(chunk, dict) and "answer" in chunk:
                answer = chunk["answer"]
                if hasattr(answer, "content"):
                    yield answer.content
                else:
                    yield str(answer)

    return StreamingResponse(generate_response(), media_type="text/plain; charset=utf-8")
