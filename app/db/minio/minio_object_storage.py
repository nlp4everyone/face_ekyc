# Minio component
from minio import Minio
from minio.datatypes import Bucket
from minio.error import S3Error
from minio.helpers import ObjectWriteResult
# Typing
from typing import List, Union
# Other component
import numpy as np
from io import BytesIO
# Image
from app.utils.image import ImagePreprocess

class MinioObjectStorage:
    def __init__(self,
                 endpoint: str,
                 bucket_name :str,
                 access_key: str | None = None,
                 secret_key: str | None = None,
                 **kwargs):
        # Params
        self._bucket_name = bucket_name
        # Init service
        self._minio_service = Minio(endpoint = endpoint,
                                    access_key = access_key,
                                    secret_key = secret_key,
                                    secure = False,
                                    **kwargs)
        # Create bucket if not existed!
        self.create_bucket()

    @property
    def list_buckets(self)-> List[Bucket]:
        return self._minio_service.list_buckets()

    def create_bucket(self,**kwargs):
        # Create bucket while not existed
        if not self._minio_service.bucket_exists(self._bucket_name):
            self._minio_service.make_bucket(bucket_name = self._bucket_name,
                                            **kwargs)

    def upload_image(self,
                     image :Union[np.ndarray,BytesIO],
                     image_name :str,
                     **kwargs) -> Union[ObjectWriteResult,None]:
        """Upload the image to MinIO"""

        # When file existed!
        if self.file_exists(image_name):
            return None
        # Convert to bytes if image is numpy array
        if isinstance(image, np.ndarray):
            # Convert image under numpy to bytes
            image = ImagePreprocess.convert_image_to_bytes(image)
        # Upload to bucket
        result = self._minio_service.put_object(bucket_name = self._bucket_name,
                                                object_name = image_name,
                                                data = image,
                                                length = len(image.getvalue()) ,
                                                content_type = "image/jpeg",
                                                **kwargs)
        return result

    def file_exists(self,
                    obj_name :str):
        """Check whether object existed in bucket or not"""
        try:
            self._minio_service.stat_object(self._bucket_name, obj_name)
            return True
        except S3Error as err:
            return False

    def remove_image(self,
                     image_name :str):
        """Check image existed before removing"""
        if self.file_exists(image_name):
            # Delete the object
            try:
                self._minio_service.remove_object(self._bucket_name, image_name)
            except S3Error as err:
                print("Error occurred while deleting object:", err)