import logging
from typing import List, Optional, Dict, Any
from injector import inject
from langchain_core.documents import Document

from src.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RagService:
    """
    RAG Service - Orchestrates retrieval for the LLM.
    
    This is a thin wrapper around RetrievalService that provides:
    - A clean interface for the LLM to get relevant chunks
    - Optional file filtering
    - Statistics and status
    """
    
    @inject
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval = retrieval_service
        logger.info("✅ RagService initialized")
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        file_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve relevant chunks for a query.
        
        This is the main entry point for the LLM to get context.
        
        Args:
            query: User query
            k: Number of chunks to return
            file_ids: Optional list of file IDs to filter by
        
        Returns:
            List of relevant Document chunks (reranked)
        """
        return self.retrieval.retrieve(query=query, k=k, file_ids=file_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        return {
            "ingestion": self.retrieval.ingestion.get_stats(),
            "retrieval": self.retrieval.get_stats()
        }