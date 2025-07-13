resource "aws_kinesis_stream" "weather_stream" {
  name             = "${var.project}-stream"
  shard_count      = 1
  retention_period = 24
}

resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.project}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "weather_predictor" {
  function_name  = "${var.project}-lambda"
  s3_bucket     = var.s3_bucket
  s3_key        = var.s3_key
  runtime        = "python3.10"
  handler        = "lambda_function.lambda_handler"   # ← file.function
  role           = aws_iam_role.lambda_exec_role.arn
  timeout        = 30
}


resource "aws_lambda_event_source_mapping" "kinesis_trigger" {
  event_source_arn = aws_kinesis_stream.weather_stream.arn
  function_name    = aws_lambda_function.weather_predictor.arn
  starting_position = "LATEST"
  batch_size       = 1
}
