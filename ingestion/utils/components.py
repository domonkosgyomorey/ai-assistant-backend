from typing import Tuple

from langchain.embeddings.base import Embeddings
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

from ingestion.config import config


def get_embeddings() -> Tuple[Embeddings, int]:
    """
    Returns the embedding and the embedding dimension.
    """

    model = VertexAIEmbeddings(
        model_name=config.embedding,
    )
    embedding_dim = len(model.embed_query("test"))
    return model, embedding_dim
