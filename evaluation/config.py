import os


class GCP:
    llm: str = "gemini-2.0-flash-001"
    embedding: str = "gemini-embedding-001"
    evaluation_bucket = "ai-assistant-evaluation"
    evaluation_docs_folder = "docs"
    evaluation_results_folder = "eval_files"

    testset_file_path = "testset.jsonl"
    knowledge_base_cache_file = "knowledge_base_cache.pkl"


class Paths:
    knowledge_base_cache_path: str = os.path.join(os.getcwd(), "knowledge_base_cache.pkl")
    evaluation_docs_cache_path: str = os.path.join(os.getcwd(), "evaluation_docs_cache.pkl")
    testset_file_path: str = os.path.join(os.getcwd(), "testset.jsonl")


class Config:
    gcp = GCP()
    paths = Paths()
