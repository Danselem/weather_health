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
  subnet_ids             = module.vpc.public_subnet_ids
  security_group_ids     = [module.vpc.alb_security_group_id]
  assign_public_ip_tasks = true

  cpu           = 512
  memory        = 1024
  desired_count = var.environment == "prod" ? 3 : 1
  min_capacity  = var.environment == "prod" ? 2 : 1
  max_capacity  = var.environment == "prod" ? 10 : 5

  db_password = var.db_password

  tags = {
    Project     = var.project
    Environment = var.environment
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = module.ecr.repository_url
}

output "s3_artifacts_bucket" {
  description = "S3 artifacts bucket name"
  value       = module.s3_artifacts.bucket_name
}

output "ecs_cluster_name" {
  description = "ECS Cluster name"
  value       = module.ecs_fargate.cluster_name
}

output "ecs_service_name" {
  description = "ECS Service name"
  value       = module.ecs_fargate.service_name
}

output "alb_dns_name" {
  description = "ALB DNS name for accessing the service"
  value       = module.ecs_fargate.alb_dns_name
}