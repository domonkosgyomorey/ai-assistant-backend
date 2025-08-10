from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from utils.config import Config


def get_llm():
    return LlamaCpp(model_path=Config.LLM_PATH, temperature=0.1, max_tokens=256, top_p=0.95, n_ctx=8192, verbose=True)


def get_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
