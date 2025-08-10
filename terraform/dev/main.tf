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
    "secretmanager.googleapis.com"   # Add this line
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

resource "google_project_service" "secret_manager" {
  project = "propane-will-468518-d0"
  service = "secretmanager.googleapis.com"
}

# Service account for GitHub Actions
resource "google_service_account" "github_sync_sa" {
  account_id   = "github-sync-bucket-reader"
  display_name = "GitHub Actions - Sync Pipeline Bucket Reader"
}

# Give it read access to the bucket
resource "google_storage_bucket_iam_member" "bucket_reader" {
  bucket = google_storage_bucket.knowledge_docs.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:github-sync-bucket-reader@propane-will-468518-d0.iam.gserviceaccount.com"
}

# Create a key for the service account
resource "google_service_account_key" "github_sync_sa_key" {
  service_account_id = google_service_account.github_sync_sa.name
}

# Store the key JSON in Secret Manager
resource "google_secret_manager_secret" "github_sync_sa_secret" {
  secret_id = "github-sync-sa-key"

  replication {
    auto {}
  }

}

resource "google_secret_manager_secret_version" "github_sync_sa_secret_version" {
  secret      = google_secret_manager_secret.github_sync_sa_secret.id
  secret_data = google_service_account_key.github_sync_sa_key.private_key
}


resource "google_secret_manager_secret_iam_member" "github_sync_sa_secret_access" {
  secret_id = "projects/propane-will-468518-d0/secrets/MONGO_URI"
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:github-sync-bucket-reader@propane-will-468518-d0.iam.gserviceaccount.com"
}