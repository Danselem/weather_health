import os
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
from prefect_aws import AwsCredentials, S3Bucket
from pydantic import SecretStr

load_dotenv(Path("./.env"))

AWS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

print("AWS_KEY_ID:", AWS_KEY_ID)
print("AWS_SECRET_KEY:", AWS_SECRET_KEY)
print("S3_BUCKET_NAME:", S3_BUCKET_NAME)

if AWS_KEY_ID is None or AWS_SECRET_KEY is None or S3_BUCKET_NAME is None:
    raise ValueError(
        "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET_NAME environment variables must be set."
    )


def create_aws_creds_block():
    if AWS_SECRET_KEY is None:
        raise ValueError(
            "AWS_SECRET_ACCESS_KEY environment variable must be set and not None."
        )
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=AWS_KEY_ID, aws_secret_access_key=SecretStr(AWS_SECRET_KEY)
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    if S3_BUCKET_NAME is None:
        raise ValueError(
            "AWS_S3_BUCKET_NAME environment variable must be set and not None."
        )
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(bucket_name=S3_BUCKET_NAME, credentials=aws_creds)
    my_s3_bucket_obj.save(name="s3-bucket", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
