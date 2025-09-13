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
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])

  project = "propane-will-468518-d0"
  service = each.value
}

resource "google_storage_bucket" "knowledge_docs" {
  name                        = "ai-assistant-dev-docs"
  location                    = "us-central1"
  project                     = "propane-will-468518-d0"
  force_destroy               = false
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

# Bucket for evaluation documents
resource "google_storage_bucket" "evaluation_docs" {
  name                        = "ai-assistant-evaluation"
  location                    = "us-central1"
  project                     = "propane-will-468518-d0"
  force_destroy               = false
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
    usage       = "evaluation-docs"
  }
}

# Bucket for public document access
resource "google_storage_bucket" "public_docs" {
  name                        = "ai-assistant-public-docs"
  location                    = "us-central1"
  project                     = "propane-will-468518-d0"
  force_destroy               = false
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
    usage       = "public-docs"
  }
}

resource "google_project_service" "secret_manager" {
  project = "propane-will-468518-d0"
  service = "secretmanager.googleapis.com"
}

# Service account for GitHub Actions
resource "google_service_account" "github_sa" {
  account_id   = "github-sa"
  display_name = "GitHub Actions SA"
}

# Give it read/write access to all buckets
resource "google_storage_bucket_iam_member" "bucket_admin_knowledge" {
  bucket = google_storage_bucket.knowledge_docs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:github-sa@propane-will-468518-d0.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "bucket_admin_evaluation" {
  bucket = google_storage_bucket.evaluation_docs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:github-sa@propane-will-468518-d0.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "bucket_admin_public" {
  bucket = google_storage_bucket.public_docs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:github-sa@propane-will-468518-d0.iam.gserviceaccount.com"
}

# Make the public docs bucket readable by anyone
resource "google_storage_bucket_iam_member" "public_docs_public_read" {
  bucket = google_storage_bucket.public_docs.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Create a key for the service account
resource "google_service_account_key" "github_sa_key" {
  service_account_id = google_service_account.github_sa.name
}

# Store the key JSON in Secret Manager
resource "google_secret_manager_secret" "github_sa_secret" {
  secret_id = "github-sa-key"

  replication {
    auto {}
  }

}

resource "google_secret_manager_secret_version" "github_sa_secret_version" {
  secret      = google_secret_manager_secret.github_sa_secret.id
  secret_data = google_service_account_key.github_sa_key.private_key
}


resource "google_secret_manager_secret_iam_member" "mongo_access" {
  secret_id = "projects/propane-will-468518-d0/secrets/MONGO_URI"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:github-sa@propane-will-468518-d0.iam.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "tavily_access" {
  secret_id = "projects/propane-will-468518-d0/secrets/TAVILY_API_KEY"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:github-sa@propane-will-468518-d0.iam.gserviceaccount.com"
}

resource "google_project_iam_member" "github_vertex_ai" {
  project = "propane-will-468518-d0"
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:github-sa@propane-will-468518-d0.iam.gserviceaccount.com"
}