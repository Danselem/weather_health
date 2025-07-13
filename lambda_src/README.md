## Build & push the image

```bash
aws ecr create-repository --repository-name weather-disease-lambda
docker build -t weather-disease-lambda .
docker tag  weather-disease-lambda:latest  <account>.dkr.ecr.<region>.amazonaws.com/weather-disease-lambda:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/weather-disease-lambda:latest
```

## Create a Kinesis Data Stream
```bash
aws kinesis create-stream \
    --stream-name weatherDiseaseStream \
    --shard-count 1
```


## Deploy the Lambda function
```bash
aws lambda create-function \
  --function-name weather-disease-infer \
  --package-type Image \
  --code ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/weather-disease-lambda:latest \
  --role arn:aws:iam::<account>:role/lambda-kinesis-role \
  --timeout 30 --memory-size 1024
```

## Wire Kinesis to Lambda (event source mapping)
```bash
aws lambda create-event-source-mapping \
  --function-name weather-disease-infer \
  --event-source-arn arn:aws:kinesis:<region>:<account>:stream/weatherDiseaseStream \
  --starting-position LATEST \
  --batch-size 100
```

## Produce test events
```bash
aws kinesis put-record \
  --stream-name weatherDiseaseStream \
  --partition-key test-1 \
  --data "$(cat sample_input.json | base64)"
```

## Watch Lambda logs:
```bash
aws logs tail /aws/lambda/weather-disease-infer --follow
```


STREAM=$(terraform output -raw stream_name)   # weather-disease-pipeline-stream

aws kinesis put-record \
  --stream-name "$STREAM" \
  --partition-key test1 \
  --data "$(base64 < sample_input.json)" \
  --region us-east-1


aws logs tail /aws/lambda/weather-disease-pipeline-lambda --follow --region us-east-1
