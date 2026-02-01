import os
import sys

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM

sys.path.append(os.path.join(os.getcwd(), "server"))
from core.utils.components import get_embedding as _get_embedding
from core.utils.components import get_llm as _get_llm


def get_embeddings() -> Embeddings:
    return _get_embedding()


def get_llm():
    return _get_llm(temperature=0)


embeddings: Embeddings = get_embeddings()
llm: LLM = get_llm()
