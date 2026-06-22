from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class MessageHandler(ABC):
    """Base interface for all message handlers"""
    
    @abstractmethod
    def get_event_type(self) -> str:
        """Return the event type this handler processes (defined by the service)"""
        pass
    
    @abstractmethod
    async def handle(self, topic: str, message: Dict[str, Any]) -> None:
        """Handle the message"""
        pass


class MessageBus(ABC):
    """Message bus interface for registering handlers and consuming messages"""
    
    @abstractmethod
    def register_handler(self, handler: MessageHandler) -> None:
        """Register a handler for a specific event type"""
        pass
    
    @abstractmethod
    def get_handler(self, event_type: str) -> Optional[MessageHandler]:
        """Get handler for event type"""
        pass