from datetime import datetime
import uuid
from sqlmodel import SQLModel, Field


class BaseModel(SQLModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class User(BaseModel, table=True):
    __tablename__ = "users"

    name: str = Field(nullable=False, max_length=50)
    username: str = Field(nullable=False, max_length=50)
    password: str = Field(nullable=False, max_length=50)


class UploadedFile(BaseModel, table=True):
    __tablename__ = "uploaded_files"

    name: str
    path: str
    category: str
    status: str


class Conversation(BaseModel, table=True):
    __tablename__ = "conversations"

    user_id: uuid.UUID = Field(
        foreign_key="users.id",
        nullable=False,
        ondelete="CASCADE"
    )


class Dialogue(BaseModel, table=True):
    __tablename__ = "dialogues"

    prompt: str = Field()
    answer: str

    conversation_id: uuid.UUID = Field(
        foreign_key="conversations.id",
        nullable=False,
        ondelete="CASCADE"
    )
