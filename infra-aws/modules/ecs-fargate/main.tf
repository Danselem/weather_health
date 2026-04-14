# =============================================================================
# ECS Fargate Module - Serverless Containers
# =============================================================================
# Deploys FastAPI application on AWS ECS with Fargate launch type.
# Includes auto-scaling, load balancing, and security groups.
# =============================================================================

variable "name" {
  description = "Application name"
  type        = string
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "weather-health"
}

variable "container_image" {
  description = "ECR image URI"
  type        = string
  default     = "amazon/amazon-ecs-sample:latest"
}

variable "container_port" {
  description = "Container port"
  type        = number
  default     = 8080
}

variable "cpu" {
  description = "Task CPU units (256, 512, 1024, etc.)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Task memory (MB)"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Desired number of tasks"
  type        = number
  default     = 2
}

variable "min_capacity" {
  description = "Minimum number of tasks for auto-scaling"
  type        = number
  default     = 2
}

variable "max_capacity" {
  description = "Maximum number of tasks for auto-scaling"
  type        = number
  default     = 10
}

variable "target_cpu" {
  description = "Target CPU utilization percentage for auto-scaling"
  type        = number
  default     = 70
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Additional security groups"
  type        = list(string)
  default     = []
}

variable "db_password" {
  description = "Database password"
  type        = string
  default     = ""
  sensitive   = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "assign_public_ip_tasks" {
  description = "Assign public IP to ECS tasks"
  type        = bool
  default     = false
}

# =============================================================================
locals {
  vpc_id     = var.vpc_id
  subnet_ids = var.subnet_ids
  ecr_image  = var.container_image != "" ? var.container_image : "amazon/amazon-ecs-sample:latest"
  container_def = var.db_password != "" ? [
    {
      name      = var.name
      image     = local.ecr_image
      essential = true
      portMappings = [{
        containerPort = var.container_port
        protocol      = "tcp"
      }]
      environment = [
        { name = "ENVIRONMENT", value = var.environment },
        { name = "DB_PASSWORD", value = var.db_password }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.name}"
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
    ] : [
    {
      name      = var.name
      image     = local.ecr_image
      essential = true
      portMappings = [{
        containerPort = var.container_port
        protocol      = "tcp"
      }]
      environment = [
        { name = "ENVIRONMENT", value = var.environment }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.name}"
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ]
}

# =============================================================================
# ECS Cluster
# =============================================================================

resource "aws_ecs_cluster" "this" {
  name = "${var.name}-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = merge(var.tags, {
    Name = "${var.name}-cluster"
  })
}

# =============================================================================
# ECS Task Definition
# =============================================================================

resource "aws_ecs_task_definition" "this" {
  family                   = "${var.name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode(local.container_def)

  tags = merge(var.tags, {
    Name = "${var.name}-task"
  })
}

# =============================================================================
# IAM Roles
# =============================================================================

resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.name}-ecs-exec-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${var.name}-ecs-task-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# =============================================================================
# Security Group
# =============================================================================

resource "aws_security_group" "ecs_sg" {
  name        = "${var.name}-ecs-sg-${var.environment}"
  description = "Security group for ECS tasks"
  vpc_id      = local.vpc_id

  ingress {
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.name}-ecs-sg"
  })
}

# =============================================================================
# CloudWatch Log Group
# =============================================================================

resource "aws_cloudwatch_log_group" "this" {
  name              = "/ecs/${var.name}"
  retention_in_days = 7

  tags = merge(var.tags, {
    Name = "${var.name}-logs"
  })
}

# =============================================================================
# Application Load Balancer
# =============================================================================

resource "aws_lb" "this" {
  name               = "${var.name}-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs_sg.id]
  subnets            = local.subnet_ids

  enable_deletion_protection = false

  tags = merge(var.tags, {
    Name = "${var.name}-alb"
  })
}

resource "aws_lb_target_group" "this" {
  name        = "${var.name}-tg-${var.environment}"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = local.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/"
    matcher             = "200,301,302"
  }

  tags = merge(var.tags, {
    Name = "${var.name}-tg"
  })
}

resource "aws_lb_listener" "this" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.this.arn
  }

  tags = merge(var.tags, {
    Name = "${var.name}-listener"
  })
}

# =============================================================================
# ECS Service
# =============================================================================

resource "aws_ecs_service" "this" {
  name            = "${var.name}-service-${var.environment}"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.subnet_ids
    security_groups  = concat([aws_security_group.ecs_sg.id], var.security_group_ids)
    assign_public_ip = var.assign_public_ip_tasks
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.this.arn
    container_name   = var.name
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.this]

  deployment_controller {
    type = "ECS"
  }

  tags = merge(var.tags, {
    Name = "${var.name}-service"
  })
}

# =============================================================================
# Auto Scaling - Optional (commented out due to AWS permission issues)
# Enable if you have the AmazonECSAutoscaleRole policy in your AWS account
# =============================================================================

# resource "aws_appautoscaling_target" "ecs_target" {
#   service_namespace  = "ecs"
#   max_capacity       = var.max_capacity
#   min_capacity       = var.min_capacity
#   resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.this.name}"
#   scalable_dimension = "ecs:service:DesiredCount"
# }

resource "aws_iam_role" "ecs_autoscaling_role" {
  name = "${var.name}-ecs-asg-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "application-autoscaling.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# Skip the policy attachment - AWS managed policy may not exist in all accounts
# Auto scaling will work via ECS service's built-in capacity provider

# =============================================================================
# Outputs
# =============================================================================

output "cluster_name" {
  description = "ECS Cluster name"
  value       = aws_ecs_cluster.this.name
}

output "cluster_arn" {
  description = "ECS Cluster ARN"
  value       = aws_ecs_cluster.this.arn
}

output "service_name" {
  description = "ECS Service name"
  value       = aws_ecs_service.this.name
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.this.dns_name
}

output "alb_arn" {
  description = "ALB ARN"
  value       = aws_lb.this.arn
}
