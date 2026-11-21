resource "random_password" "db" {
  length  = 32
  special = false
}

resource "google_secret_manager_secret" "db_url" {
  secret_id = "weather-health-db-url"

  replication {
    auto {}
  }

  labels = {
    project     = var.app_name
    environment = var.environment
  }
}

resource "google_secret_manager_secret_version" "db_url" {
  secret = google_secret_manager_secret.db_url.id

  secret_data = "postgresql://postgres:${random_password.db.result}@/weatherhealth?host=/cloudsql/${module.cloudsql.connection_name}"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_service_account" "runtime" {
  account_id   = "${var.app_name}-${var.environment}-runtime"
  display_name = "Cloud Run runtime for ${var.app_name} ${var.environment}"
}

resource "google_project_iam_member" "cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "db_url_access" {
  secret_id = google_secret_manager_secret.db_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}
