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