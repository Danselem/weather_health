# GCP Infrastructure

Terraform for the Cloud Run deployment of the weather-health API.

> **Live demo:** https://weather-health-587752925399.us-central1.run.app/docs
>
> The live resources were initially created with `gcloud`. The Terraform
> configuration now models them, but adopting the existing resources requires
> imports (see [docs/GCP_DEPLOYMENT.md](../docs/GCP_DEPLOYMENT.md)).

## Architecture

```text
Artifact Registry -> Cloud Run -> Cloud SQL Postgres
                          |-> Secret Manager (PREDICTION_DB_URL)
                          |-> GCS artifacts bucket (private)
                          |-> Cloud Logging
```

- Cloud Run service `weather-health`, region `us-central1`, max 5 instances.
- Dedicated runtime service account with `roles/cloudsql.client` and
  `roles/secretmanager.secretAccessor`.
- Cloud SQL Postgres `weather-health-db`, database `weatherhealth`.
- Secret Manager secret `weather-health-db-url`.
- GCS artifacts bucket with uniform access and no public IAM.
- Artifact Registry repository `weather-health`.

## Modules

| Module | Purpose |
|---|---|
| `modules/cloudrun` | Cloud Run service, env vars, secret refs, Cloud SQL annotation |
| `modules/cloudsql` | Postgres instance and database |
| `modules/artifact-registry` | Docker image registry |
| `modules/gcs-artifacts` | Private artifact bucket |

## Quick start

```bash
gcloud auth application-default login
cd infra-gcp
terraform init
terraform plan -var-file=environments/dev/terraform.tfvars
terraform apply -var-file=environments/dev/terraform.tfvars
```

The root module uses local state today. Add a GCS backend before using this as
a team-owned stack.

## Adopting the live demo

The live Cloud SQL instance, secret, Artifact Registry, and Cloud Run service
were created manually. Import them before applying this configuration:

```bash
cd infra-gcp

terraform import module.cloudsql.google_sql_database_instance.this weather-health-db
terraform import module.cloudsql.google_sql_database.this \
  projects/scenic-dynamo-447811-m9/instances/weather-health-db/databases/weatherhealth
terraform import google_secret_manager_secret.db_url \
  projects/scenic-dynamo-447811-m9/secrets/weather-health-db-url
terraform import google_secret_manager_secret_version.db_url \
  projects/scenic-dynamo-447811-m9/secrets/weather-health-db-url/versions/1
terraform import module.artifact_registry.google_artifact_registry_repository.this \
  projects/scenic-dynamo-447811-m9/locations/us-central1/repositories/weather-health
terraform import module.cloudrun.google_cloud_run_service.this \
  locations/us-central1/namespaces/scenic-dynamo-447811-m9/services/weather-health
```

Imports are stateful and should be done in a branch with a reviewed plan. The
secret version uses `ignore_changes = [secret_data]` so an import does not
rotate the live password.

## Security notes

- The GCS bucket is private; the previous `allUsers` viewer binding was removed.
- The Cloud Run service is public for the demo (`allUsers` invoker).
- Cloud SQL uses a public IP with the Cloud Run connector; no authorized
  network is opened.
- Automated backups are disabled on the demo instance. Enable them and set
  deletion protection before production.
- Store secrets only in Secret Manager, never in tfvars or Git.
