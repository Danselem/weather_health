# main.tf (optional)
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

module "streaming" {
  source    = "./modules/streaming"
  project   = var.project
  s3_bucket = aws_s3_bucket.lambda_code.id
  s3_key    = aws_s3_object.lambda_zip.key
}

