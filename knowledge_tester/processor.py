import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import joblib
from chunker.recursive_chunker import RecursiveChunker
from cluster_descriptor.tfidf_descriptor import TfidfDiscriminativeDescriptor
from document_clustering.document_clustering import EmbeddedDocument
from document_clustering.hdbscan_document_clustering import HDBSCANDocumentClustering
from resource_loader.docx_loader import DocxLoader
from resource_loader.pdf_loader import PDFLoader
from resource_loader.txt_loader import TxtLoader
from resource_loader.web_loader import WebLoader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from utils.common_types import Cluster, CustomDocument, custom_document_to_document, document_to_custom_document
from utils.components import embeddings

LOADERS = {
    ".pdf": PDFLoader,
    ".txt": TxtLoader,
    ".docx": DocxLoader,
}


def process_documents(
    file_paths: list[str], ignored_keywords: list[str], separate_clustering: bool = False
) -> list[Cluster]:
    """Process documents with either separate or combined clustering strategy.

    Args:
        file_paths: List of file paths or URLs to process
        ignored_keywords: Keywords to exclude from TF-IDF analysis
        separate_clustering: If True, cluster each file separately; if False, combine all documents

    Returns:
        List of clusters with documents and descriptions
    """
    mode = "🔄 Separate clustering" if separate_clustering else "🔗 Combined clustering"
    print(
        f"\n{'=' * 60}\n"
        f"📚 DOCUMENT PROCESSING - {mode}\n"
        f"Files: {len(file_paths)} | Ignored keywords: {len(ignored_keywords)}\n"
        f"{'=' * 60}"
    )

    if separate_clustering:
        return _process_documents_separately(file_paths, ignored_keywords)
    else:
        return _process_documents_combined(file_paths, ignored_keywords)


def _get_file_cache_key(file_path: str) -> str:
    """Generate a unique cache key based on file path, modification time, and size."""
    file_stat = os.stat(file_path)
    file_info = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
    return hashlib.md5(file_info.encode()).hexdigest()


def _cache_embeddings_for_file(file_path: str, document_texts: list[str], source_type: str) -> list[list[float]]:
    """Load embeddings from cache if available, otherwise generate and cache them.

    Args:
        file_path: Path to the file being processed
        document_texts: List of document text contents
        source_type: Type of source ('file' or 'url')

    Returns:
        List of embedding vectors
    """
    if source_type == "file":
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        cache_key = _get_file_cache_key(file_path)
        cache_file = cache_dir / f"embeddings_{cache_key}.joblib"

        if cache_file.exists():
            cached_embeddings = joblib.load(cache_file)
            if len(cached_embeddings) == len(document_texts):
                print(f"✅ Using cached embeddings for {os.path.basename(file_path)}")
                return cached_embeddings

    print(f"🔄 Generating embeddings for {os.path.basename(file_path) if source_type == 'file' else 'URL'}...")
    embs = embeddings.embed_documents(document_texts)

    if source_type == "file":
        joblib.dump(embs, cache_file)

    return embs


def _is_url(path: str) -> bool:
    """Check if the given path is a valid URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def chunking(document: list[CustomDocument]) -> list[CustomDocument]:
    """Chunk documents into smaller pieces.

    Args:
        document: List of CustomDocument to chunk

    Returns:
        List of chunked CustomDocument
    """
    if not document:
        return document

    docs = [custom_document_to_document(doc) for doc in document]
    chunked_docs = RecursiveChunker.chunk(docs)
    result = []
    for chunk_doc in chunked_docs:
        result.append(document_to_custom_document(chunk_doc))
    return result


def _load_documents_from_file(file_path: str) -> Tuple[list[CustomDocument], str]:
    """Load documents from a file or URL.

    Args:
        file_path: Path to file or URL to load

    Returns:
        Tuple of (list of documents, status message)
    """
    try:
        ext = Path(file_path).suffix.lower()
        loader_class = LOADERS.get(ext)

        if not loader_class:
            if _is_url(file_path):
                loader_class = WebLoader
            else:
                return [], "❌ Unsupported format"

        documents = loader_class.load(file_path)
        if not documents:
            return [], "❌ No content"

        valid_documents = [doc for doc in documents if doc["content"].strip()]
        if valid_documents:
            valid_documents = chunking(valid_documents)
            return valid_documents, "✅ Loaded"
        else:
            return [], "❌ No valid text"

    except Exception as e:
        error_msg = str(e)
        return [], f"❌ Error: {error_msg}..."


def _prepare_stop_words(ignored_keywords: list[str]) -> list[str]:
    """Combine English stop words with user-provided ignored keywords."""
    return [word.lower() for word in list(ENGLISH_STOP_WORDS.union(set(ignored_keywords)))]


def _create_clusters_with_hdbscan(
    documents: list[CustomDocument], embeddings_list: list[list[float]], stop_words: list[str]
) -> list[Cluster]:
    """Cluster documents using HDBSCAN algorithm with TF-IDF descriptors.

    Args:
        documents: List of documents to cluster
        embeddings_list: List of embedding vectors for documents
        stop_words: Words to exclude from TF-IDF analysis

    Returns:
        List of clusters with descriptions
    """
    extended_docs = [
        EmbeddedDocument(document=doc, embedding=emb) for doc, emb in zip(documents, embeddings_list, strict=True)
    ]
    return HDBSCANDocumentClustering().cluster(extended_docs, TfidfDiscriminativeDescriptor(stop_words=stop_words))


def _create_fallback_cluster(file_path: str, documents: list[CustomDocument]) -> Cluster:
    """Create a single cluster containing all documents when clustering fails.

    Args:
        file_path: Path to the source file
        documents: List of documents to include in cluster

    Returns:
        A single cluster containing all documents
    """
    return {
        "id": str(uuid.uuid4()),
        "description": f"Fallback cluster from {os.path.basename(file_path)}",
        "source": file_path,
        "documents": documents,
    }


def _print_processing_summary(file_stats: list[dict], title: str = "Processing Summary") -> None:
    """Print a formatted summary table of file processing results.

    Args:
        file_stats: List of dictionaries containing file statistics
        title: Title for the summary table
    """
    print(f"\n📊 {title}:")
    print("─" * 80)

    has_clusters = any("clusters" in stat for stat in file_stats)

    if has_clusters:
        header = f"{'File':<30} {'Chunks':<8} {'Clusters':<10} {'Status'}"
    else:
        header = f"{'File':<30} {'Chunks':<8} {'Status'}"

    print(header)
    print("─" * 80)

    totals = {"chunks": 0, "clusters": 0}

    for stat in file_stats:
        chunks = stat.get("chunks", 0)
        totals["chunks"] += chunks

        if has_clusters:
            clusters = stat.get("clusters", 0)
            totals["clusters"] += clusters
            print(f"{stat['name'][:29]:<30} {chunks:<8} {clusters:<10} {stat['status']}")
        else:
            print(f"{stat['name'][:29]:<30} {chunks:<8} {stat['status']}")

    print("─" * 80)

    final_status = "🎯 Complete" if has_clusters else "📚 Ready for clustering"

    if has_clusters:
        summary = f"TOTAL{'':<25} {totals['chunks']:<8} {totals['clusters']:<10} {final_status}"
    else:
        summary = f"TOTAL{'':<25} {totals['chunks']:<8} {final_status}"

    print(summary)
    print("─" * 80)


def _process_documents_separately(file_paths: list[str], ignored_keywords: list[str]) -> list[Cluster]:
    """Process each file separately and cluster documents within each file.

    Args:
        file_paths: List of file paths to process
        ignored_keywords: Keywords to exclude from TF-IDF analysis

    Returns:
        List of all clusters from all files
    """
    stop_words = _prepare_stop_words(ignored_keywords)
    print(f"\n🔄 Processing {len(file_paths)} files in parallel...")

    def process_single_file(file_path: str) -> Tuple[list[Cluster], str, dict]:
        """Process a single file and return its clusters and statistics."""
        file_name = os.path.basename(file_path)
        stats = {"name": file_name, "chunks": 0, "clusters": 0, "status": "❌ Failed"}

        documents, status = _load_documents_from_file(file_path)
        if not documents:
            stats["status"] = status
            return [], file_path, stats

        stats["chunks"] = len(documents)

        document_texts = [doc["content"] for doc in documents]
        file_type = documents[0]["source_type"]
        embs = _cache_embeddings_for_file(file_path, document_texts, source_type=file_type)

        try:
            file_clusters = _create_clusters_with_hdbscan(documents, embs, stop_words)

            for cluster in file_clusters:
                cluster["source"] = file_path
                cluster["description"] = f"[{file_name}] {cluster['description']}"

            stats["clusters"] = len(file_clusters)
            stats["status"] = "✅ Clustered"
            return file_clusters, file_path, stats

        except Exception as e:
            # Fallback to single cluster if clustering fails
            single_cluster = _create_fallback_cluster(file_path, documents)
            stats["clusters"] = 1
            stats["status"] = f"⚠️  Fallback: {str(e)[:20]}..."
            return [single_cluster], file_path, stats

    all_clusters = []
    file_stats = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file_path): file_path for file_path in file_paths}

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_clusters, _, stats = future.result()
                file_stats.append(stats)
                all_clusters.extend(file_clusters)

            except Exception as e:
                file_name = os.path.basename(file_path)
                file_stats.append(
                    {"name": file_name, "chunks": 0, "clusters": 0, "status": f"❌ Exception: {str(e)[:20]}..."}
                )

    _print_processing_summary(file_stats)

    if not all_clusters:
        raise ValueError("❌ No valid documents found. Please check your files.")

    return all_clusters


def _process_documents_combined(file_paths: list[str], ignored_keywords: list[str]) -> list[Cluster]:
    """Process all files together and cluster the combined document set.

    Args:
        file_paths: List of file paths to process
        ignored_keywords: Keywords to exclude from TF-IDF analysis

    Returns:
        List of clusters from combined documents
    """
    print(f"\n🔗 Processing {len(file_paths)} files as combined dataset...")

    all_documents = []
    file_stats = []

    for file_path in file_paths:
        documents, status = _load_documents_from_file(file_path)
        all_documents.extend(documents)
        file_stats.append({"name": os.path.basename(file_path), "chunks": len(documents), "status": status})

    _print_processing_summary(file_stats, "File Loading Summary")

    if not all_documents:
        raise ValueError("❌ No valid documents found. Please check your files.")

    document_texts = [doc["content"] for doc in all_documents]
    print(f"\n🔗 Embedding {len(document_texts)} documents as single batch...")
    embs = embeddings.embed_documents(document_texts)
    print("✅ Embedding completed")

    print(f"🔍 Clustering {len(all_documents)} combined documents...")
    stop_words = _prepare_stop_words(ignored_keywords)

    clusters = _create_clusters_with_hdbscan(all_documents, embs, stop_words)

    for cluster in clusters:
        cluster["source"] = "combined"
        cluster["description"] = f"[Combined] {cluster['description']}"

    print(f"✅ Found {len(clusters)} clusters from combined dataset")
    return clusters
