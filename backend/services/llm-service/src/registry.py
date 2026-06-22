"""
Handler Registry - Registers all handlers for this service
"""

import logging
from common.message_bus.interfaces import MessageBus #type: ignore
from src.dependencies import injector, EVENT_HANDLER_MAP

logger = logging.getLogger(__name__)


def register_handlers(bus: MessageBus):
    """Register all handlers on the message bus"""
    print("===========================================", flush=True)

    for event_class, handler_class in EVENT_HANDLER_MAP.items():
        
        bus.register(event_class, handler_class)  # ✅ pass class
        logger.info(f"Registered: {event_class.event_type()} -> {handler_class.__name__}")


def get_consumption_topics(bus: MessageBus) -> list:
    """Return topics to consume based on registered handlers"""
    topics = []
    
    for event_class, handler_class in EVENT_HANDLER_MAP.items():
        topic = event_class.topic()
        topics.append(topic)
        logger.debug(f"   Event '{event_class.event_type()}' -> topic '{topic}'")
    
    logger.info(f"📡 Consumption topics: {topics}")
    return topics


def get_event_types() -> dict:
    """Return mapping of event_type to event_class"""
    event_types = {}
    for event_class, handler_class in EVENT_HANDLER_MAP.items():
        event_types[event_class.event_type()] = event_class
    return event_types