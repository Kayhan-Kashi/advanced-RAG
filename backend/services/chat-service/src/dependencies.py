from __future__ import annotations
import os
import logging

from injector import Binder, Module, SingletonScope, Injector
from common.kafka.producer import KafkaProducer, get_producer #type: ignore
from common.events import DocumentEmbeddingDoneEvent, PromptAnswerCompletedEvent #type: ignore
from handlers.document_embedding_done_handler import DocumentEmbeddingDoneHandler
from handlers.prompt_answer_completed_handler import PromptAnswerCompletedHandler
from services.conversation_service import ConversationService
from services.document_service import DocumentService

logger = logging.getLogger(__name__)

def get_kafka_producer() -> KafkaProducer:
    """Factory function for Kafka producer"""
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    logger.info(f"Creating Kafka producer with bootstrap: {bootstrap_servers}")
    return get_producer(bootstrap_servers=bootstrap_servers)


EVENT_HANDLER_MAP = {
    DocumentEmbeddingDoneEvent: DocumentEmbeddingDoneHandler,
    PromptAnswerCompletedEvent: PromptAnswerCompletedHandler,
}


class DependencyInjection(Module):
    """Dependency injection configuration"""
    
    def configure(self, binder: Binder):
        binder.bind(KafkaProducer, to=get_kafka_producer, scope=SingletonScope)
        binder.bind(ConversationService, scope=SingletonScope)
        binder.bind(DocumentService, scope=SingletonScope)  
        
        binder.bind(DocumentEmbeddingDoneHandler, scope=SingletonScope)
        binder.bind(PromptAnswerCompletedHandler, scope=SingletonScope)


injector = Injector([DependencyInjection()])