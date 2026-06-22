from abc import ABC, abstractmethod
from datetime import datetime
import uuid
from pydantic import BaseModel


class BaseEvent(BaseModel, ABC):
    event_id: str = str(uuid.uuid4())
    timestamp: datetime = datetime.utcnow()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    
    @classmethod
    @abstractmethod
    def event_type(cls) -> str:
        """Get the event type - must be overridden as class method"""
        pass
    
    @classmethod
    @abstractmethod
    def topic(cls) -> str:
        """Get the topic - must be overridden as class method"""
        pass
    
    def model_dump(self, *args, **kwargs):
        """Override to include event_type and topic in serialization"""
        data = super().model_dump(*args, **kwargs)
        data['event_type'] = self.__class__.event_type()
        data['topic'] = self.__class__.topic()
        return data