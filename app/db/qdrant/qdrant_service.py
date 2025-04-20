# Qdrant client
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models.models import UpdateResult
# Typing
from typing import Union
import uuid
# Face Request
from app.core.schema import FaceRequest
# Exception
from app.core.exceptions import (UserExistedException,
                                 UserNotFoundException)

class QdrantService:
    def __init__(self,
                 embedding_dims :int,
                 host :str = "localhost",
                 port :int = 6333,
                 collection_name :str = "face_embeddings",
                 distance :Distance = Distance.COSINE):
        # Init client
        self._client = AsyncQdrantClient(host = host,
                                         port = port)
        # Init params
        self._embedding_dims = embedding_dims
        self._collection_name = collection_name
        self._distance_metric = distance

    async def create_collection(self):
        """Create new collection if not existed"""
        status = await self._client.collection_exists(self._collection_name)
        # Create collection (if not exists)
        if not status:
            await self._client.create_collection(
                collection_name = self._collection_name,
                vectors_config = VectorParams(size = self._embedding_dims,
                                              distance = self._distance_metric)
            )

    async def insert_face_embedding(self,
                                    face_request :FaceRequest) -> UpdateResult:
        # Define key field
        filter = Filter(
            must=[
                FieldCondition(
                    key = "face_id",
                    match = MatchValue(value = face_request.face_id)
                )
            ]
        )
        # Search with limit=1 to quickly check if any match exists
        point_appearance = await self._client.scroll(collection_name = self._collection_name,
                                                     scroll_filter = filter,
                                                     limit = 1)

        # Check if point exist on system or not
        point_appeared = len(point_appearance[0]) > 0
        # When point existed, raise a exception
        if point_appeared:
            raise UserExistedException(user_id = face_request.face_id)

        # Index points with vectors and payloads
        points = [
            PointStruct(
                id = str(uuid.uuid4()),
                vector = face_request.embedding,
                payload = face_request.model_dump(exclude={"embedding"})
            ),
        ]
        return await self._client.upsert(collection_name = self._collection_name,
                                         points = points)

    async def _get_point_id(self,
                            face_id :str):
        # Define the filter based on the payload condition
        payload_filter = Filter(
            must = [
                FieldCondition(key = "face_id",
                               match = MatchValue(value = face_id))]
        )

        # Result
        result = await self._client.scroll(collection_name = self._collection_name,
                                           scroll_filter = payload_filter,
                                           limit = 1,
                                           with_payload = True,
                                           with_vectors = False)
        return result[0][0] if len(result[0]) >0 else None

    async def delete_point(self,
                           face_id :str):
        # Get point
        searched_point = await self._get_point_id(face_id)
        # Check if point valid
        if searched_point is None:
            raise UserNotFoundException(user_id = face_id)

        # Delete the point
        result = await self._client.delete(collection_name = self._collection_name,
                                           points_selector = [searched_point.id])
        return searched_point.payload, result


