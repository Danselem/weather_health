

# New bucket for Lambda deployment packages
resource "aws_s3_bucket" "lambda_code" {
  bucket = "${var.project}-lambda-code-bucket"

  tags = {
    Name        = "${var.project}-lambda-code-bucket"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "lambda_code_block" {
  bucket = aws_s3_bucket.lambda_code.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "lambda_code_versioning" {
  bucket = aws_s3_bucket.lambda_code.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Upload your zipped lambda code to the bucket
resource "aws_s3_object" "lambda_zip" {
  bucket = aws_s3_bucket.lambda_code.id
  key    = "app.zip"                        # or whatever you name your zip
  source = "modules/streaming/lambda/app.zip"          # relative path to your zipped code
  etag   = filemd5("modules/streaming/lambda/app.zip")
}
