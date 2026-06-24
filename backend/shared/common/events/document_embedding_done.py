from .base_event import BaseEvent


class DocumentEmbeddingDoneEvent(BaseEvent):
    document_id: str
    filetype: str
    filename: str
    status: str
    
    
    @classmethod
    def event_type(cls) -> str:
        return "document_embedding_done"
    
    @classmethod
    def topic(cls) -> str:
        return "document-embedding-done"