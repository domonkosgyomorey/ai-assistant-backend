import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from giskard.llm import set_embedding_model, set_llm_model
from giskard.llm.client.base import ChatMessage
from giskard.rag.evaluate import QATestset, RAGReport, evaluate
from giskard.rag.testset_generation import generate_testset
from langchain_core.documents import Document

from evaluation.config import Config
from evaluation.utils.custom_metric_generator import create_heatmap, create_low_performance_bar_chart
from evaluation.utils.document_processor import DocumentProcessorImpl
from evaluation.utils.gcp_helper import GCPHelper
from evaluation.utils.knowlede_communication import call
from evaluation.utils.persistent_knowledge_base import PersistentKnowledgeBase
from evaluation.utils.utils import get_embedding, get_llm


class LLMAdapter:
    def __init__(self):
        self.client = get_llm()

    def complete(self, messages: list[ChatMessage], **kwargs) -> ChatMessage:
        formatted_messages = "\n".join(
            ["User: " + m.content if m.role == "user" else "AI: " + m.content for m in messages]
        )
        response = self.client.invoke(formatted_messages, **kwargs)
        content = response.content if hasattr(response, "content") else str(response)
        return ChatMessage(role="ai", content=content)


class EmbeddingAdapter:
    def __init__(self):
        self.embedding = get_embedding()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(self.embedding.embed(texts))


class GiskardHelper:
    def __init__(self, load_knowledge_from_cache: bool = False):
        self.evaluation_docs_bucket = GCPHelper(
            bucket_name=Config.gcp.evaluation_bucket, prefix=Config.gcp.evaluation_docs_folder
        )
        self.evaluation_results_bucket = GCPHelper(
            bucket_name=Config.gcp.evaluation_bucket, prefix=Config.gcp.evaluation_results_folder
        )

        set_llm_model(Config.gcp.llm)
        set_embedding_model(Config.gcp.embedding)

        self.llm = LLMAdapter()
        self.embedding = EmbeddingAdapter()
        self.persistent_kb = PersistentKnowledgeBase(
            data=self._documents_to_dataframe(self._load_documents()),
            columns=["source", "page_content"],
            llm_client=self.llm,
            embedding_model=self.embedding,
            cache_path=Config.paths.knowledge_base_cache_path,
        )

        if load_knowledge_from_cache:
            self.evaluation_results_bucket.download_file(
                Config.gcp.knowledge_base_cache_file, Config.paths.knowledge_base_cache_path
            )
            self.persistent_kb.load_from_cache(Config.paths.knowledge_base_cache_path)

        self.knowledge_base = self.persistent_kb.get_knowledge_base()
        print(f"Knowledge base initialized with {len(self.knowledge_base.topics)} topics.")

    def testset_generation(
        self,
        number_of_samples: int,
        agent_description: str,
        output_path: str | None = None,
        upload_results: bool = False,
    ) -> QATestset:
        testset: QATestset = generate_testset(
            self.knowledge_base, number_of_samples, agent_description=agent_description
        )

        if output_path:
            testset.save(output_path)

        if upload_results:
            self._save_and_upload_testset(testset)

        self.persistent_kb.save_to_cache(Config.paths.knowledge_base_cache_path)
        self.evaluation_results_bucket.upload_file(
            Config.paths.knowledge_base_cache_path, Config.gcp.knowledge_base_cache_file
        )

        return testset

    def testset_evaluation(self, upload_results: bool = False):
        self.evaluation_results_bucket.download_file(Config.gcp.testset_file_path, Config.paths.testset_file_path)
        testset: QATestset = QATestset.load(Config.paths.testset_file_path)
        result: RAGReport = evaluate(
            call,
            testset,
            self.knowledge_base,
            agent_description="This agent can answer questions about University of Obuda in topics such as students' requirements, administration and etc.",
        )
        local_prefix = Path.cwd().as_posix() + "/temp/"
        Path(local_prefix).mkdir(parents=True, exist_ok=True)
        eval_report_path = local_prefix + "evaluation_report.json"
        html_report_path = local_prefix + "evaluation_report.html"
        metrics_path = local_prefix + "metrics.txt"
        heatmap_path = local_prefix + "heatmap.png"
        low_perf_path = local_prefix + "low_performance.png"

        result.to_pandas().to_json(eval_report_path, orient="records", lines=True)
        result.to_html(html_report_path)

        result_str = result.component_scores().to_string()
        result_str += "\n\n"
        result_str += result.correctness_by_question_type().to_string()
        result_str += "\n\n"
        result_str += result.correctness_by_topic().to_string()
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(result_str)

        create_heatmap(result.to_pandas(), heatmap_path)
        create_low_performance_bar_chart(result.component_scores().reset_index(), low_perf_path)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        gcp_prefix = f"report/{timestamp}/"
        if upload_results:
            self.evaluation_results_bucket.upload_file(eval_report_path, gcp_prefix + "evaluation_report.json")
            self.evaluation_results_bucket.upload_file(html_report_path, gcp_prefix + "evaluation_report.html")
            self.evaluation_results_bucket.upload_file(metrics_path, gcp_prefix + "metrics.txt")
            self.evaluation_results_bucket.upload_file(heatmap_path, gcp_prefix + "heatmap.png")
            self.evaluation_results_bucket.upload_file(low_perf_path, gcp_prefix + "low_performance.png")

    def _save_and_upload_testset(self, testset: QATestset):
        temp_path = "temp_testset.jsonl"
        testset.save(temp_path)
        try:
            self.evaluation_results_bucket.upload_file(temp_path, Config.gcp.testset_file_path)
            print(f"Testset uploaded to GCP: {Config.gcp.testset_file_path}")
        except Exception as e:
            print(f"Failed to upload testset: {e}")
            raise

    def _documents_to_dataframe(self, docs: list[Document]) -> pd.DataFrame:
        processor = DocumentProcessorImpl()
        docs: list[Document] = processor.process(docs)
        docs = [d.model_dump() for d in docs]
        df = pd.DataFrame()
        for d in docs:
            d |= {"source": d["metadata"].get("source", "unknown")}
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
        return df

    def _load_documents(self) -> list[Document]:
        cache_path = Path(Config.paths.evaluation_docs_cache_path)

        if cache_path.exists():
            try:
                print(f"Loading documents from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cached_docs = pickle.load(f)
                return cached_docs
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-downloading...")

        print("Downloading and processing documents...")
        with tempfile.TemporaryDirectory() as temp_dir:
            self.evaluation_docs_bucket.download_all(temp_dir)

            documents = []
            processed_files = 0

            pickle_files = list(Path(temp_dir).rglob("*.pkl"))
            total_pickle_files = len(pickle_files)
            print(f"Processing {total_pickle_files} pickle files...")

            for file_path in pickle_files:
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)
                    if isinstance(obj, Document):
                        documents.append(obj)
                    elif isinstance(obj, list) and all(isinstance(item, Document) for item in obj):
                        documents.extend(obj)
                    else:
                        print(f"Unrecognized object type in {file_path}: {type(obj)}")

                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files}/{total_pickle_files} pickle files")

        try:
            print(f"Caching {len(documents)} documents to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(documents, f)
            print("Documents cached successfully")
        except Exception as e:
            print(f"Failed to cache documents: {e}")

        return documents
