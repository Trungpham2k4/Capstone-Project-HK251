# services/minio_service.py
from minio import Minio
from io import BytesIO
from datetime import timedelta

class MinioService:
    def __init__(self, endpoint="localhost:9000", access_key="admin", secret_key="password", secure=False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def ensure_bucket(self, bucket):
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)
            print(f"[MinIO] Created bucket: {bucket}")

    def put_object(self, bucket: str, key: str, data_bytes: bytes):
        self.ensure_bucket(bucket)
        self.client.put_object(
            bucket_name=bucket,
            object_name=key,
            data=BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="text/plain"
        )
        print(f"[MinIO] Uploaded {bucket}/{key}")

    def get_object(self, bucket: str, key: str) -> bytes:
        response = self.client.get_object(bucket, key)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def get_presigned_url(self, bucket: str, key: str, expire_hours=1):
        return self.client.presigned_get_object(bucket, key, expires=timedelta(hours=expire_hours))