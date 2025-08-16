from core.config.config import config
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings


def get_llm(temperature=0.1, max_tokens=1024):
    return ChatVertexAI(
        model_name=config.llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embedding():
    return VertexAIEmbeddings(
        model_name=config.embedding,
    )
