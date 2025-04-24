# init model
from app.utils.face.embedding import AdaFaceEmbedding, TimmEmbedding
from app.utils.face.recognition import MTCNNRecognition
# Define service
from app.db.minio import MinioObjectStorage
from app.db.elastic_search import ElasticSearchService
# Import config
from app.core.config.constant import (EMBEDDING_MODEL,
                                      MINIO_BUCKET_NAME,
                                      FACE_EMBEDDING_DIMS)
from app.core.config import (HF_TOKEN,
                             MINIO_ACCESS_KEY,
                             MINIO_SECRET_KEY,
                             ES_USER,
                             ES_PASSWORD)

# Variable
mtcnn = None
face_embedding_model = None
minio_storage = None
qdrant_service = None
es_service = None

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

def init_elastic_search() -> ElasticSearchService:
    global es_service
    es_service = ElasticSearchService(host = "http://elasticsearch:9200",
                                      user = ES_USER,
                                      password = ES_PASSWORD,
                                      embedding_dims = FACE_EMBEDDING_DIMS)
    return es_service

def get_face_embedding_model():
    return face_embedding_model

def get_face_recognition_model() ->MTCNNRecognition:
    return mtcnn

def get_minio_storage() -> MinioObjectStorage:
    return minio_storage

def get_elastic_search() -> ElasticSearchService:
    return es_service
