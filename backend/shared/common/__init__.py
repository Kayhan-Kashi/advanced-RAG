from .events import BaseEvent, PromptAnswerRequestedEvent
from .kafka import KafkaProducer, get_producer, shutdown_producer

__version__ = "1.0.0"

__all__ = [
    "BaseEvent",
    "PromptAnswerRequestedEvent", 
    "KafkaProducer",
    "get_producer",
    "shutdown_producer",
]