from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict
from uuid import UUID


class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    id: UUID
    name: str
    category: str
    created_at: datetime
    message: str
    
    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Response model for a single document"""
    id: UUID
    name: str
    category: str
    path: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response model for list of documents"""
    total: int
    documents: List[DocumentResponse]
    
    class Config:
        from_attributes = True


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion"""
    message: str
    document_id: str


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics"""
    total_documents: int
    by_category: Dict[str, int]