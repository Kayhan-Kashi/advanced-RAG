from .base_event import BaseEvent


class DocumentUploadedEvent(BaseEvent):
    document_id: str
    filetype: str
    filename: str
    
    @classmethod
    def event_type(cls) -> str:
        return "document_uploaded"
    
    @classmethod
    def topic(cls) -> str:
        return "document-upload"