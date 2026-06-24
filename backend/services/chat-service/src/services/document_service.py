import uuid
import os
import shutil
from fastapi import HTTPException, UploadFile
from injector import inject
from sqlmodel import Session, select
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from database.models import UploadedFile
from common.kafka.producer import KafkaProducer #type: ignore
from common.events.document_uploaded import DocumentUploadedEvent #type: ignore

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document uploads"""
    @inject
    def __init__(self, kafka_producer: Optional[KafkaProducer] = None):
        self.upload_dir = os.getenv('UPLOAD_DIR', './uploads')
        self.kafka_producer = kafka_producer
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def upload_document(
        self,
        session: Session,
        file: UploadFile,
        user_id: str,
        category: str = "general"
    ) -> UploadedFile:
        """Upload a document file"""
        try:
            allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
                )
            
            file_id = uuid.uuid4()
            safe_filename = f"{file_id}{file_ext}"
            file_path = os.path.join(self.upload_dir, safe_filename)
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            document = UploadedFile(
                id=file_id,
                name=file.filename, 
                path=file_path,
                category=category,  
                status="uploaded"    
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            
            logger.info(f"✅ Document uploaded: {file.filename}")
            logger.info(f"   Document ID: {document.id}")
            logger.info(f"   File path: {document.path}")
            logger.info(f"   Status: {document.status}")
            
            if self.kafka_producer:
                doc_event = DocumentUploadedEvent(
                    document_id=str(document.id),
                    filetype=file_ext,
                    filename=document.name,
                    user_id=user_id
                )
                  
                self.kafka_producer.produce(event=doc_event, key=str(document.id))
                logger.info(f"✅ Kafka event published successfully for document: {doc_event.event_id}")
            
            return document
            
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error uploading document: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    
    def get_all_documents(
        self,
        session: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[UploadedFile]:
        """Get all documents (no user filter)"""
        try:
            documents = session.exec(
                select(UploadedFile)
                .order_by(UploadedFile.created_at.desc())
                .offset(skip)
                .limit(limit)
            ).all()
            
            logger.info(f"📋 Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")
    
    def get_document_by_id(
        self,
        session: Session,
        document_id: str
    ) -> Optional[UploadedFile]:
        """Get document by ID"""
        try:
            document = session.get(UploadedFile, uuid.UUID(document_id))
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            return document
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    def delete_document(
        self,
        session: Session,
        document_id: str
    ) -> Dict[str, Any]:
        """Delete a document"""
        try:
            # Get document
            document = session.get(UploadedFile, uuid.UUID(document_id))
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            if os.path.exists(document.path):
                os.remove(document.path)
                logger.info(f"🗑️ File deleted from disk: {document.path}")
            
            session.delete(document)
            session.commit()
            
            logger.info(f"🗑️ Document deleted: {document.name}")
            
            return {
                "message": "Document deleted successfully", 
                "document_id": document_id
            }
            
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
    
    def update_document_status(
        self,
        session: Session,
        document_id: str,
        status: str,
    ) -> Optional[UploadedFile]:
        """Update document status"""
        try:
            document = session.get(UploadedFile, uuid.UUID(document_id))
            if not document:
                logger.warning(f"⚠️ Document not found: {document_id}")
                return None
            
            document.status = status
            
            session.add(document)
            session.commit()
            session.refresh(document)
            
            logger.info(f"✅ Document {document_id} status updated to: {status}")
            return document
            
        except Exception as e:
            logger.error(f"❌ Error updating document status: {str(e)}")
            return None
    
    def get_document_stats(
        self,
        session: Session
    ) -> Dict[str, Any]:
        """Get statistics about all documents"""
        try:
            documents = session.exec(select(UploadedFile)).all()
            
            stats = {
                "total_documents": len(documents),
                "by_category": {},
                "by_status": {}
            }
            
            for doc in documents:
                cat = doc.category
                stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
                
                status = doc.status or "unknown"
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            logger.info(f"📊 Retrieved document stats: {stats['total_documents']} documents")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")