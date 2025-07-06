import os
import boto3
import pika
import json
from urllib.parse import urlparse

def list_s3_files(s3_uri):
    """Lists all files in the given S3 URI directory."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'):  # Skip directories
                files.append(f"s3://{bucket}/{key}")
    return files

def publish_to_rabbitmq(rabbitmq_url, queue_name, messages):
    """Publishes a list of messages to a RabbitMQ queue."""
    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)

    for message in messages:
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)  # make message persistent
        )
        print(f"Published: {message['frame_id']}")

    connection.close()

def generate_messages_from_s3(s3_uri, video_id, parent_job_id=None):
    """Generates RabbitMQ-compatible job messages from an S3 directory."""
    s3_files = list_s3_files(s3_uri)
    messages = []

    for s3_path in s3_files:
        filename = os.path.basename(s3_path)
        frame_id = filename
        frame_number = os.path.splitext(filename)[0]

        message = {
            "job_id": f"job-{frame_number}",
            "video_id": video_id,
            "frame_id": frame_id,
            "frame_number": frame_number,
            "s3_frame_path": s3_path,
        }
        if parent_job_id:
            message["parent_job_id"] = parent_job_id

        messages.append(message)

    return messages

if __name__ == "__main__":
    # Example usage
    S3_URI = "s3://rww-projects/t6HXEjJGvrx9/HFWHq9rwUrat/safety_video/1/images/"
    VIDEO_ID = "1"
    PARENT_JOB_ID = "parent-job-001"  # Optional
    RABBITMQ_URL = "amqps://queue_user:buzwok-6hadci-heSkec@b-81da8f52-16dc-4859-a8c9-ede56c07d4b0.mq.us-east-1.amazonaws.com:5671"
    QUEUE_NAME = "safety-tracking-video-instance-segmentation"

    messages = generate_messages_from_s3(S3_URI, VIDEO_ID, PARENT_JOB_ID)
    publish_to_rabbitmq(RABBITMQ_URL, QUEUE_NAME, messages)
