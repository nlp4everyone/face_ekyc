# Elastic Search
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
# Typing
from typing import Literal, Optional, List
# Config
from app.core.config.constant import EMBEDDING_COLLECTION_NAME
# Face Request
from app.core.schema import (FaceRequest,
                             RequestResult,
                             ResponseStatus,
                             RequestType,
                             RetrievedResult)
# Base Model
from app.core.vector_store import BaseVectorStore
# Other component
import uuid
# Exception
from app.core.exceptions import (UserExistedException,
                                 UserNotFoundException)
from loggers import SystemLogger

class ElasticSearchService(BaseVectorStore):
    def __init__(self,
                 host :str,
                 user :str,
                 password :str,
                 embedding_dims: int,
                 index_name :str = EMBEDDING_COLLECTION_NAME,
                 metric_type :Literal["cosine","l2_norm","dot_product"] = "cosine"):
        super().__init__(embedding_dims = embedding_dims)
        # Define client
        self._client = AsyncElasticsearch(hosts = [host],
                                          basic_auth = (user, password),
                                          verify_certs = False)
        # Params
        self._index_name = index_name
        self._metric_type = metric_type

    async def create_index(self):
        """Create new collection if not existed"""
        # Check if index is existed or not
        if not await self._client.indices.exists(index = self._index_name):
            await self._client.indices.create(index = self._index_name, body={
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self._embedding_dims,
                            "index": True,
                            "similarity": self._metric_type
                        },
                        "payload": {
                            "type": "object"
                        }
                    }
                }
            })

    async def _get_document_by_id(self,
                                  face_id :str):
        """Delete point from collection"""
        query = {
            "query": {
                "term": {
                    "payload.face_id.keyword": face_id
                }
            }
        }
        # Searched point
        searched_point = await self._client.search(index=self._index_name,
                                                   body=query,
                                                   size=1)
        # When document not existed
        if searched_point["hits"]["total"]["value"] == 0:
            return None
        return searched_point["hits"]

    async def insert_face_embedding(self,
                                    face_request :FaceRequest,
                                    doc_id: Optional[str] = None) -> RequestResult:
        """Insert new face information to Elastic Search Database"""
        # Check if user existed or not
        existed_document = await self._get_document_by_id(face_id = face_request.face_id)
        if existed_document is not None:
            # Raise exception if existed
            raise UserExistedException(user_id = face_request.face_id)

        # When id is empty
        if doc_id is None: doc_id = str(uuid.uuid4())
        # Index
        status = await self._client.index(index = self._index_name,
                                          id = doc_id,
                                          document = {"embedding": face_request.embedding,
                                                      "payload": face_request.model_dump(exclude={"embedding"})})
        # Return
        return RequestResult(request_id = str(uuid.uuid4()),
                             status = ResponseStatus.COMPLETED,
                             request_type = RequestType.INSERT)

    async def retrieve_points(self,
                              embedding: List[float],
                              similarity_top_k: int = 3) -> List[RetrievedResult]:
        """Retrive similar point in Qdrant from inputing value"""
        # Define value
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": embedding,
                "k": similarity_top_k,
                "num_candidates": 100
            },
            "_source": True
        }

        results = []
        try:
            # Search
            response = await self._client.search(index = self._index_name,
                                                 body = query)
            # Return response
            if response and "hits" in response and "hits" in response["hits"]:
                for hit in response["hits"]["hits"]:
                    # Get payload value
                    source = hit["_source"]
                    # Score
                    score = dict(hit).get("_score")
                    payload = source.get("payload")
                    # Append
                    if payload is not None:
                        results.append(RetrievedResult(score = float(score),
                                                       payload = payload))
            return results
        except NotFoundError:
            return results

    async def delete_point(self, face_id: str) -> RequestResult:
        """Delete point from collection"""
        query = {
            "query": {
                "term": {
                    "payload.face_id.keyword": face_id
                }
            }
        }

        # Check if user existed or not
        existed_document = await self._get_document_by_id(face_id = face_id)
        if existed_document is None:
            # Raise exception if not existed
            raise UserNotFoundException(user_id = face_id)

        # Delete documents matching the query
        response = await self._client.delete_by_query(index = self._index_name,
                                                      body = query)
        # Return
        return RequestResult(request_id = str(uuid.uuid4()),
                             status = ResponseStatus.COMPLETED,
                             request_type = RequestType.DELETE,
                             data = existed_document["hits"][0]["_source"]["payload"])

    async def close_connection(self):
        await self._client.close()
