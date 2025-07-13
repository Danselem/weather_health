# Allow Lambda to read from the Kinesis stream
resource "aws_iam_role_policy" "lambda_kinesis_read" {
  name = "${var.project}-lambda-kinesis-read"
  role = aws_iam_role.lambda_exec_role.id   # the role you already created

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = [
        "kinesis:GetRecords",
        "kinesis:GetShardIterator",
        "kinesis:DescribeStream",
        "kinesis:DescribeStreamSummary",
        "kinesis:ListShards",
        "kinesis:ListStreams"
      ],
      Resource = aws_kinesis_stream.weather_stream.arn
    }]
  })
}
