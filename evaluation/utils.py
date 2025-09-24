from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

from evaluation.config import Config


def get_llm(temperature=0.1, max_tokens=1024):
    return ChatVertexAI(
        model_name=Config.gcp.llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embedding():
    vertex_embeddings = VertexAIEmbeddings(
        model_name=Config.gcp.embedding,
    )
    return vertex_embeddings
