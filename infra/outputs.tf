# existing outputs …
output "ec2_public_ip" {
  value = aws_instance.ubuntu.public_ip
}

output "rds_endpoint" {
  value = aws_db_instance.mlflow_pg.endpoint
}

# NEW — forward streaming module outputs
output "stream_name" {
  value = module.streaming.stream_name
}

output "lambda_name" {
  value = module.streaming.lambda_name
}


