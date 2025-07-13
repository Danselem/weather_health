output "stream_name" {
  value = aws_kinesis_stream.weather_stream.name
}

output "lambda_name" {
  value = aws_lambda_function.weather_predictor.function_name
}
