from fastapi import APIRouter, Depends
from fastapi_injector import Injected
from sqlmodel import Session
from starlette import status
from starlette.exceptions import HTTPException
from typing import List, Optional

from api.schemas.conversation_schemas import (
    CreateConversationSchema, 
    CreateDialogueSchema,
    UpdateDialogueSchema,
)
from database.sqlite_session import get_session
from services.conversation_service import ConversationService
from api.websocket.manager import connection_manager

router = APIRouter(
    prefix="/conversation",
    tags=["conversation"]
)


# ---------------------------------------------------------
# CREATE CONVERSATION
# ---------------------------------------------------------
@router.post(
    "/new",
    status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    data: CreateConversationSchema,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Create a new conversation"""
    conversation_id = service.create_conversation(
        session=db,
        user_id=data.user_id 
    )
    
    # Register the conversation for WebSocket routing
    # connection_manager.register_conversation(
    #     conversation_id=str(conversation_id),
    #     user_id=data.user_id
    # )
    
    return {
        "conversation_id": conversation_id,
        "message": "Conversation created successfully"
    }


# ---------------------------------------------------------
# CREATE DIALOGUE
# ---------------------------------------------------------
@router.post(
    "/{conversation_id}/dialogue",
    status_code=status.HTTP_201_CREATED
)
async def create_dialogue(
    conversation_id: str,
    data: CreateDialogueSchema,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Create a new dialogue in a conversation"""
    dialogue_id = await service.create_dialogue(
        session=db,
        conversation_id=conversation_id,
        prompt=data.prompt,
        answer=data.answer if hasattr(data, 'answer') else None
    )

    return {
        "dialogue_id": dialogue_id,
        "message": f"Dialogue created successfully for conversation: {conversation_id}"
    }


# ---------------------------------------------------------
# GET CONVERSATION BY ID
# ---------------------------------------------------------
@router.get(
    "/{conversation_id}",
    response_model=dict
)
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Get a conversation with all its dialogues"""
    conversation = service.get_conversation(
        session=db,
        conversation_id=conversation_id
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    
    return conversation


# ---------------------------------------------------------
# GET USER CONVERSATIONS
# ---------------------------------------------------------
@router.get(
    "/user/{user_id}",
    response_model=List[dict]
)
async def get_user_conversations(
    user_id: str,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Get all conversations for a specific user"""
    return service.get_user_conversations(session=db, user_id=user_id)


# ---------------------------------------------------------
# UPDATE DIALOGUE ANSWER
# ---------------------------------------------------------
@router.put(
    "/dialogue/{dialogue_id}/answer",
    status_code=status.HTTP_200_OK
)
async def update_dialogue_answer(
    dialogue_id: str,
    data: UpdateDialogueSchema,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Update a dialogue with the assistant's answer"""
    dialogue = service.update_dialogue_answer(
        session=db,
        dialogue_id=dialogue_id,
        answer=data.answer
    )
    
    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dialogue not found",
        )
    
    return {
        "dialogue_id": dialogue.id,
        "answer": dialogue.answer,
        "message": "Answer added successfully"
    }


# ---------------------------------------------------------
# DELETE CONVERSATION
# ---------------------------------------------------------
@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK
)
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Delete a conversation and all its dialogues"""
    result = service.delete_conversation(
        session=db,
        conversation_id=conversation_id
    )
    
    # Clean up WebSocket routing
    if conversation_id in connection_manager.conversation_routing:
        del connection_manager.conversation_routing[conversation_id]
    
    return result


# ---------------------------------------------------------
# GET CONVERSATION DIALOGUES (PAGINATED)
# ---------------------------------------------------------
@router.get(
    "/{conversation_id}/dialogues",
    response_model=List[dict]
)
async def get_conversation_dialogues(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Get dialogues for a conversation with pagination"""
    return service.get_conversation_dialogues(
        session=db, 
        conversation_id=conversation_id,
        limit=limit,
        offset=offset
    )


# ---------------------------------------------------------
# GET DIALOGUE BY ID
# ---------------------------------------------------------
@router.get(
    "/dialogue/{dialogue_id}",
    response_model=dict
)
async def get_dialogue_by_id(
    dialogue_id: str,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService),
):
    """Get a single dialogue by ID"""
    dialogue = service.get_dialogue_by_id(
        session=db,
        dialogue_id=dialogue_id
    )
    
    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dialogue not found",
        )
    
    return dialogue