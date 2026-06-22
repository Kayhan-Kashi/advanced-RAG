from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi_injector import Injected
from sqlmodel import Session
from starlette import status
from typing import List, Optional

from api.schemas.document_schemas import (
    UploadDocumentResponse, 
    DocumentResponse, 
    DocumentListResponse, 
    DeleteDocumentResponse, 
    DocumentStatsResponse
)
from database.sqlite_session import get_session
from services.document_service import DocumentService

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)


@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),  # Keep for upload (passed to Kafka)
    db: Session = Depends(get_session),
    service: DocumentService = Injected(DocumentService),
):
    """Upload a single document"""
    document = await service.upload_document(
        session=db,
        file=file,
        user_id=user_id
    )
    
    return UploadDocumentResponse(
        id=document.id,
        name=document.name,
        category=document.category,
        created_at=document.created_at,
        message="Document uploaded successfully"
    )


@router.get("/", response_model=DocumentListResponse)
async def get_all_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    db: Session = Depends(get_session),
    service: DocumentService = Injected(DocumentService),
):
    """Get all documents (no user filter)"""
    documents = service.get_all_documents(
        session=db,
        skip=skip,
        limit=limit
    )
    
    return DocumentListResponse(
        total=len(documents),
        documents=[
            DocumentResponse(
                id=doc.id,
                name=doc.name,
                category=doc.category,
                path=doc.path,
                created_at=doc.created_at
            )
            for doc in documents
        ]
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_session),
    service: DocumentService = Injected(DocumentService),
):
    """Get a specific document by ID"""
    document = service.get_document_by_id(
        session=db,
        document_id=document_id
    )
    
    return DocumentResponse(
        id=document.id,
        name=document.name,
        category=document.category,
        path=document.path,
        created_at=document.created_at
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    db: Session = Depends(get_session),
    service: DocumentService = Injected(DocumentService),
):
    """Delete a document"""
    result = service.delete_document(
        session=db,
        document_id=document_id
    )
    return DeleteDocumentResponse(
        message=result["message"],
        document_id=result["document_id"]
    )


@router.get("/stats/summary", response_model=DocumentStatsResponse)
async def get_document_stats(
    db: Session = Depends(get_session),
    service: DocumentService = Injected(DocumentService),
):
    """Get document statistics"""
    stats = service.get_document_stats(session=db)
    return DocumentStatsResponse(
        total_documents=stats["total_documents"],
        by_category=stats["by_category"]
    )