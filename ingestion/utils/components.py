from typing import Tuple

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings


def get_embeddings() -> Tuple[Embeddings, int]:
    """
    Returns the embedding and the embedding dimension.
    """

    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_dim = len(model.embed_query("test"))
    return model, embedding_dim
