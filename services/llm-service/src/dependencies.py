from __future__ import annotations
import os

from injector import Binder, Module, SingletonScope, Injector
from common.kafka.producer import KafkaProducer, get_producer #type: ignore
from common.events import PromptAnswerRequestedEvent, DocumentUploadedEvent #type: ignore
from src.services.llm_service import LLMService
from src.core.embedding_model import EmbeddingModel
from src.core.document_loader import DocumentLoader
from src.core.text_chunker import TextChunker
from src.core.text_preprocessor import TextPreprocessor
from src.services.rag_service import RagService
from src.services.ingestion_service import IngestionService
from src.services.retrieval_service import RetrievalService
from src.handlers.document_uploaded_handler import DocumentUploadedHandler
from src.handlers.prompt_answer_requested_handler import PromptAnswerRequestedHandler


def get_kafka_producer() -> KafkaProducer:
    """Factory function for Kafka producer"""
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    return get_producer(bootstrap_servers=bootstrap_servers)


# ============================================
# Event to Handler Mapping
# ============================================

EVENT_HANDLER_MAP = {
    PromptAnswerRequestedEvent: PromptAnswerRequestedHandler,
    DocumentUploadedEvent: DocumentUploadedHandler,
}


class DependencyInjection(Module):
    """Dependency injection configuration for LLM service"""
    
    def configure(self, binder: Binder):
        # Bind services
        binder.bind(KafkaProducer, to=get_kafka_producer, scope=SingletonScope)
        binder.bind(LLMService, scope=SingletonScope)
        binder.bind(EmbeddingModel, scope=SingletonScope)
        binder.bind(DocumentLoader, scope=SingletonScope)
        binder.bind(TextPreprocessor, scope=SingletonScope)
        binder.bind(TextChunker, scope=SingletonScope)
        binder.bind(IngestionService, scope=SingletonScope)
        binder.bind(RetrievalService, scope=SingletonScope)
        binder.bind(RagService, scope=SingletonScope)
        
        binder.bind(DocumentUploadedHandler, scope=SingletonScope)
        binder.bind(PromptAnswerRequestedHandler, scope=SingletonScope)


injector = Injector([DependencyInjection()])