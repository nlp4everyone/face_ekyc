# Qdrant client
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
# Typing
from typing import List
import uuid
# Face Request
from app.core.schema import (FaceRequest,
                             RequestResult,
                             ResponseStatus,
                             RequestType,
                             RetrievedResult)
# Exception
from app.core.exceptions import (UserExistedException,
                                 UserNotFoundException)
# Base Model
from app.core.vector_store import BaseVectorStore

class QdrantService(BaseVectorStore):
    def __init__(self,
                 embedding_dims :int,
                 host :str = "localhost",
                 port :int = 6333,
                 collection_name :str = "face_embeddings",
                 distance :Distance = Distance.COSINE):
        super().__init__(embedding_dims = embedding_dims)
        # Init client
        self._client = AsyncQdrantClient(host = host,
                                         port = port)
        # Init params
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
                                    face_request :FaceRequest) -> RequestResult:
        """Insert new face information to Qdrant Database"""
        # Get point
        point_appeared = await self._get_point_id(face_request.face_id,
                                                  limit = 1)

        # When point existed, raise an exception
        if point_appeared is not None:
            raise UserExistedException(user_id = face_request.face_id)

        # Index points with vectors and payloads
        points = [
            PointStruct(
                id = str(uuid.uuid4()),
                vector = face_request.embedding,
                payload = face_request.model_dump(exclude={"embedding"})
            ),
        ]
        # Update
        status = await self._client.upsert(collection_name = self._collection_name,
                                           points = points)
        # Return
        return RequestResult(request_id = str(uuid.uuid4()),
                             status = ResponseStatus.COMPLETED,
                             request_type = RequestType.INSERT)

    async def _get_point_id(self,
                            face_id :str,
                            limit :int = 1):
        """Get point information from defined field (face id)"""
        # Define the filter based on the payload condition
        payload_filter = Filter(
            must = [
                FieldCondition(key = "face_id",
                               match = MatchValue(value = face_id))]
        )

        # Result
        result = await self._client.scroll(collection_name = self._collection_name,
                                           scroll_filter = payload_filter,
                                           limit = limit,
                                           with_payload = True,
                                           with_vectors = False)
        # Return
        return result[0][0] if len(result[0]) >0 else None

    async def delete_point(self,
                           face_id :str) -> RequestResult:
        """Delete specified point in Qdrant with field (face id)"""
        # Get point
        searched_point = await self._get_point_id(face_id)
        # Check if point valid
        if searched_point is None:
            raise UserNotFoundException(user_id = face_id)

        # Delete the point
        result = await self._client.delete(collection_name = self._collection_name,
                                           points_selector = [searched_point.id])
        # Return
        return RequestResult(request_id = str(uuid.uuid4()),
                             status = ResponseStatus.COMPLETED,
                             request_type = RequestType.DELETE,
                             data = searched_point.payload)

    async def retrieve_points(self,
                              embedding :List[float],
                              similarity_top_k :int = 3) -> List[RetrievedResult]:
        """Retrive similar point in Qdrant from inputing value"""
        searched_results = await self._client.search(collection_name = self._collection_name,
                                                     query_vector = embedding,
                                                     limit = similarity_top_k)
        # Empty response
        if len(searched_results) == 0:
            return []
        # Retrieved points
        return [RetrievedResult(score = result.score,
                                payload = result.payload) for result in searched_results]