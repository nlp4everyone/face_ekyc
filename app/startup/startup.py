# init model
from app.utils.face.embedding import AdaFaceEmbedding
from app.utils.face.recognition import MTCNNRecognition
from app.db.minio import MinioObjectStorage
# Import config
from app.core.config.constant import EMBEDDING_MODEL, MINIO_BUCKET_NAME
from app.core.config import (HF_TOKEN,
                             MINIO_ACCESS_KEY,
                             MINIO_SECRET_KEY,
                             MINIO_PORT)

# Variable
mtcnn = None
ada_face = None
minio_storage = None

def init_models():
    """Start Postgres Connection"""
    global mtcnn
    global ada_face
    # Init connection
    mtcnn = MTCNNRecognition()
    ada_face = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
                                HF_TOKEN = HF_TOKEN)
    return mtcnn, ada_face

def init_minio_storage():
    global minio_storage
    minio_storage = MinioObjectStorage(endpoint = "minio:9000",
                                       bucket_name = MINIO_BUCKET_NAME,
                                       access_key = MINIO_ACCESS_KEY,
                                       secret_key = MINIO_SECRET_KEY)

def get_face_embedding_model():
    return ada_face

def get_face_recognition_model():
    return mtcnn

def get_minio_storage():
    return minio_storage
