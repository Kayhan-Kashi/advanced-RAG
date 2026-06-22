from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import uuid

class CreateConversationSchema(BaseModel):
    user_id: str

class CreateDialogueSchema(BaseModel):
    prompt: str
    answer: Optional[str] = None

class UpdateDialogueSchema(BaseModel):
    answer: str

class DialogueResponseSchema(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    prompt: str
    answer: Optional[str]
    created_at: datetime
    updated_at: datetime

class ConversationResponseSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    dialogues: List[DialogueResponseSchema]

class UserConversationsSchema(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    title: str
    dialogue_count: int