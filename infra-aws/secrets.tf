# Generated database password. The secret value is sensitive in Terraform
# state, so remote state must be encrypted and access-controlled.
resource "random_password" "db" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_secretsmanager_secret" "db" {
  name                    = "${var.project}-${var.environment}-db-password"
  description             = "Database password for ${var.project} ${var.environment}"
  recovery_window_in_days = 7

  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_secretsmanager_secret_version" "db" {
  secret_id     = aws_secretsmanager_secret.db.id
  secret_string = random_password.db.result
}
