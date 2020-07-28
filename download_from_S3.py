#!/usr/bin/env python3
# modified from macaw/local/download_from_S3.py

import argparse
import boto3


def download_from_S3():
    parser = argparse.ArgumentParser(description="This script downloads punctuation model from an S3 bucket.")
    parser.add_argument("AWS_ACCESS_KEY_ID", help="AWS_ACCESS_KEY_ID")
    parser.add_argument("AWS_SECRET_ACCESS_KEY", help="AWS_SECRET_ACCESS_KEY")
    parser.add_argument("s3_bucket", help="S3 bucket")
    parser.add_argument("s3_key", help="S3 key")
    parser.add_argument("dst", help="Destination")
    args = parser.parse_args()

    boto3.setup_default_session(aws_access_key_id=args.AWS_ACCESS_KEY_ID, aws_secret_access_key=args.AWS_SECRET_ACCESS_KEY)

    s3_client = boto3.client("s3")
    s3_client.download_file(args.s3_bucket, args.s3_key, args.dst)


if __name__ == "__main__":
    download_from_S3()
