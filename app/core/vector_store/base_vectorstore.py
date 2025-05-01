# Typing
from typing import List
import uuid
# Face Request
from app.core.schema import (FaceRequest,
                             RequestResult,
                             RetrievedResult)

class BaseVectorStore:
    def __init__(self,
                 embedding_dims :int):
        # Init params
        self._embedding_dims = embedding_dims

    async def insert_face_embedding(self,
                                    face_request :FaceRequest) -> RequestResult:
        """Insert new face information to Qdrant Database"""
        raise NotImplementedError()

    async def delete_point(self,
                           face_id :str) -> RequestResult:
        """Delete specified point in Qdrant with field (face id)"""
        # Get point
        raise NotImplementedError()

    async def retrieve_points(self,
                              embedding :List[float],
                              similarity_top_k :int = 3) -> List[RetrievedResult]:
        """Retrive similar point in Qdrant from inputing value"""
        raise NotImplementedError()