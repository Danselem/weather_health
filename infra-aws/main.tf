# =============================================================================
# Main Terraform Configuration
# =============================================================================
# Root module that combines all modules for weather-health infrastructure.
# =============================================================================

module "vpc" {
  source = "./modules/vpc"

  name        = var.app_name
  environment = var.environment
  project     = var.project

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

module "ecr" {
  source = "./modules/ecr"

  name                  = var.app_name
  environment           = var.environment
  project               = var.project
  enable_scan_on_push   = true
  enable_immutable_tags = var.environment == "prod"
}

module "s3_artifacts" {
  source = "./modules/s3-artifacts"

  name        = "${var.project}-${var.environment}-artifacts"
  environment = var.environment
  project     = var.project

  lifecycle_rules = [
    {
      id              = "expire-old-models"
      expiration_days = var.environment == "prod" ? 90 : 30
    }
  ]
}

module "ecs_fargate" {
  source = "./modules/ecs-fargate"

  name            = var.app_name
  environment     = var.environment
  project         = var.project
  container_image = var.container_image
  container_port  = 8080

  vpc_id                 = module.vpc.vpc_id
  subnet_ids             = module.vpc.private_subnet_ids
  alb_subnet_ids         = module.vpc.public_subnet_ids
  security_group_ids     = []
  alb_security_group_id  = module.vpc.alb_security_group_id
  assign_public_ip_tasks = false

  cpu           = 512
  memory        = 1024
  desired_count = var.environment == "prod" ? 3 : 1
  min_capacity  = var.environment == "prod" ? 2 : 1
  max_capacity  = var.environment == "prod" ? 10 : 5

  db_secret_arn = aws_secretsmanager_secret.db.arn

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}
