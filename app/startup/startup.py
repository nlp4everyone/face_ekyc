# init model
from app.utils.face.embedding import AdaFaceEmbedding, TimmEmbedding
from app.utils.face.recognition import MTCNNRecognition
# Define service
from app.db.minio import MinioObjectStorage
from app.db.qdrant import QdrantService, Distance
# Import config
from app.core.config.constant import (EMBEDDING_MODEL,
                                      MINIO_BUCKET_NAME,
                                      FACE_EMBEDDING_DIMS)
from app.core.config import (HF_TOKEN,
                             MINIO_ACCESS_KEY,
                             MINIO_SECRET_KEY)

# Variable
mtcnn = None
face_embedding_model = None
minio_storage = None
qdrant_service = None

def init_models():
    """Start Postgres Connection"""
    global mtcnn
    global face_embedding_model
    # Init connection
    mtcnn = MTCNNRecognition(device = "cuda:0")
    # face_embedding_model = AdaFaceEmbedding(model_name = EMBEDDING_MODEL,
    #                                         HF_TOKEN = HF_TOKEN)
    face_embedding_model = TimmEmbedding(device = "cuda")
    return mtcnn, face_embedding_model

def init_minio_storage():
    global minio_storage
    minio_storage = MinioObjectStorage(endpoint = "minio:9000",
                                       bucket_name = MINIO_BUCKET_NAME,
                                       access_key = MINIO_ACCESS_KEY,
                                       secret_key = MINIO_SECRET_KEY)

def init_qdrant_service() -> QdrantService:
    global qdrant_service
    qdrant_service = QdrantService(host = "qdrant",
                                   embedding_dims = FACE_EMBEDDING_DIMS,
                                   distance = Distance.COSINE)
    return qdrant_service

def get_face_embedding_model():
    return face_embedding_model

def get_face_recognition_model():
    return mtcnn

def get_minio_storage():
    return minio_storage

def get_qdrant_service() -> QdrantService:
    return qdrant_service
