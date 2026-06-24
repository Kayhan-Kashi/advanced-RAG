import logging
from typing import Optional, Any
from injector import inject
from common.events.document_embedding_done import DocumentEmbeddingDoneEvent #type: ignore
from services.document_service import DocumentService

logger = logging.getLogger(__name__)


class DocumentEmbeddingDoneHandler:
    """Handler for DocumentEmbeddingDoneEvent - updates document status"""
    
    @inject
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
        logger.info("✅ DocumentEmbeddingDoneHandler initialized")
    
    async def handle(self, event: DocumentEmbeddingDoneEvent, db: Optional[Any] = None):
        try:
            logger.info(f"📥 [Chat] Processing document embedding done event")
            logger.info(f"   Event ID: {event.event_id[:8]}...")
            logger.info(f"   Document ID: {event.document_id[:8]}...")
            logger.info(f"   Filename: {event.filename}")
            logger.info(f"   Status: {event.status}")
            
            document = self.document_service.update_document_status(
                session=db,
                document_id=event.document_id,
                status=event.status,
            )
            
            if document:
                logger.info(f"✅ [Chat] Document status updated to: indexed")
                logger.info(f"   Document: {document.name}")
                logger.info(f"   Path: {document.path}")
            else:
                logger.warning(f"⚠️ [Chat] Document not found: {event.document_id}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [Chat] Error updating document status: {e}")
            logger.exception("Stack trace:")
            raise