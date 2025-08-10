# Ingestion Pipelines Documentation

## Overview
The ingestion pipelines process and store documents efficiently. The two main pipelines are:

1. **Knowledge Builder**: Processes documents from a local folder and stores them in MongoDB.
2. **Sync Pipeline**: Synchronizes new or updated documents from a source (e.g., GCP bucket).

> [!WARNING] 
> PDF is the only supported document currently.

---

## Knowledge Builder Pipeline

### Usage
Run the following command:
```bash
uv run -- python -m ingestion.knowledge_builder.py --source-folder <path_to_source_folder> [--collection-name <collection_name>] [--use-existing-collection] [--clear-collection-before]
```

### Key Arguments
- `--source-folder` (required): The path to the folder containing documents to be ingested. This folder should include all files you want to process.
- `--collection-name` (optional): The name of the MongoDB collection where the processed documents will be stored. If not provided, the default collection name from the configuration will be used.
- `--use-existing-collection` (optional): If specified, the pipeline will use an existing MongoDB collection. If not specified and the collection exists, the pipeline will fail.
- `--clear-collection-before` (optional): If specified, the pipeline will clear all existing data in the collection before ingesting new documents.

---

## Sync Pipeline

### Usage
Run the following command:
```bash
uv run -- python -m ingestion.sync_pipeline [--use-existing-collection] [--clear-collection-before]
```

### Key Arguments
- `--use-existing-collection` (optional): If specified, the pipeline will use an existing MongoDB collection. If not specified and the collection exists, the pipeline will fail.
- `--clear-collection-before` (optional): If specified, the pipeline will clear all existing data in the collection before ingesting new documents.

### Configuration
Ensure the following fields are correctly set in `ingestion/config.py`:
- `BUCKET_NAME`: The default GCP bucket name.
- `PROJECT_ID`: The GCP project ID.
- `COLLECTION_NAME`: The default MongoDB collection name.

---

## Preparing Files

### File Placement
- Place files in a local folder (e.g., `./pdfs`) for the Knowledge Builder.
- Ensure files are in supported formats (e.g., PDF).
- Place the pdfs to the GCP Bucket to add new documents with the Sync Pipeline.

### Folder Structure
Example:
```
pdfs/
├── document1.pdf
├── document2.pdf
└── subfolder/
    └── document3.pdf
```

---
