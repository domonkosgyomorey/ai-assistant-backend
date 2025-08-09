provider "google" {
  project = "propane-will-468518-d0"
  region  = "us-central1"
}

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

resource "google_project_service" "enable_apis" {
  for_each = toset([
    "run.googleapis.com",         # Cloud Run
    "aiplatform.googleapis.com",  # Vertex AI / Gemini
    "storage.googleapis.com",     # Cloud Storage
    "iam.googleapis.com"          # IAM
  ])

  project = "propane-will-468518-d0"
  service = each.value
}

resource "google_storage_bucket" "knowledge_docs" {
  name                        = "ai-assistant-dev-docs"
  location                    = "us-central1"
  project                     = "propane-will-468518-d0"
  force_destroy               = false  # Prevents deletion if not empty
  uniform_bucket_level_access = true

  versioning {
    enabled = false
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 365
    }
  }

  labels = {
    environment = "dev"
    usage       = "knowledge-docs"
  }
}
