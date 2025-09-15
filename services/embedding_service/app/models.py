from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class EmbedRequest(BaseModel):
    id: str = Field(..., description="The id of the request")
    text: str = Field(..., description="The text of the request")

class BulkEmbedRequest(BaseModel):
    docs: List[EmbedRequest]

class EmbedResponse(BaseModel):
    id: str = Field(..., description="The id of the response")
    embedding: List[float]

class BulkEmbedResponse(BaseModel):
    upserted: int
    processed: int

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    k: Optional[int] = Field(3, description="Number of nearest neighbors to return (default=3)")

class SearchResult(BaseModel):
    id: str
    text: Optional[str]
    metadata: Optional[Dict[str, Any]] = {}
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

class DeleteStaleRequest(BaseModel):
    days: Optional[int] = Field(30, description="Delete docs older than N days (default=30)")

class DeleteStaleResponse(BaseModel):
    deleted: int