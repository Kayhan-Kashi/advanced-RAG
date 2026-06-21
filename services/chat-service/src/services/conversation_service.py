# llm-service/src/services/conversation_service.py
import sys
import uuid
from fastapi import HTTPException
from injector import inject
from sqlmodel import Session, select
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from common.events import PromptAnswerRequestedEvent #type: ignore
from common.kafka.producer import KafkaProducer #type: ignore
from database.models import Conversation, Dialogue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


logger = logging.getLogger(__name__)


class ConversationService:
    """Service for managing conversations and dialogues"""
    
    @inject
    def __init__(self, kafka_producer: Optional[KafkaProducer] = None):
        """
        Initialize ConversationService with optional Kafka producer
        
        Args:
            kafka_producer: Injected Kafka producer (optional)
        """
        self.kafka_producer = kafka_producer
        logger.info("✅ ConversationService initialized")
    
    def create_conversation(self, session: Session, user_id: str):
        """Create a new conversation"""
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        try:
            conv = Conversation(user_id=uuid.UUID(user_id))
            session.add(conv)
            session.commit()
            session.refresh(conv)
            logger.info(f"✅ Conversation created: {conv.id} for user: {user_id}")
            return conv.id
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format for user_id")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error creating conversation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")
    
    def _get_conversation_history(self, session: Session, conversation_id: str, max_messages: int = 50) -> List[Dict[str, str]]:
        """
        Get conversation history as a list of messages.
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            max_messages: Maximum number of messages to include
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        try:
            dialogues = session.exec(
                select(Dialogue)
                .where(Dialogue.conversation_id == uuid.UUID(conversation_id))
                .order_by(Dialogue.created_at.desc())
                .limit(max_messages // 2 + 1)  
            ).all()
            
            history = []
            for dialogue in reversed(dialogues):  
                history.append({
                    "role": "user",
                    "content": dialogue.prompt
                })
                if dialogue.answer:
                    history.append({
                        "role": "assistant",
                        "content": dialogue.answer
                    })
            
            if len(history) > max_messages:
                history = history[-max_messages:]
            
            logger.debug(f"📜 Retrieved {len(history)} messages from conversation history")
            return history
            
        except Exception as e:
            logger.warning(f"Failed to get conversation history: {e}")
            return []
    
    async def create_dialogue(
        self,
        session: Session,
        conversation_id: str,
        prompt: str,
        answer: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
        include_history: bool = True
    ):
        """Create a new dialogue in a conversation and publish event to Kafka
        
        Args:
            session: Database session
            conversation_id: ID of the conversation
            prompt: User's prompt/question
            answer: Optional answer (for existing dialogues)
            file_ids: Optional list of document IDs to restrict search context
            include_history: Whether to include conversation history in the event
        """
        if not conversation_id:
            raise HTTPException(status_code=400, detail="Conversation ID is required")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        try:
            # Check if conversation exists
            conversation = session.get(Conversation, uuid.UUID(conversation_id))
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Create dialogue in database
            dialogue = Dialogue(
                conversation_id=uuid.UUID(conversation_id),
                prompt=prompt,
                answer=answer if answer else ""
            )
            session.add(dialogue)
            session.commit()
            session.refresh(dialogue)
            
            logger.info(f"📝 Dialogue created: {dialogue.id} for conversation: {conversation_id}")
            if file_ids:
                logger.info(f"   Associated file_ids: {file_ids}")
            
            history = None
            if include_history:
                history = self._get_conversation_history(session, conversation_id)
                if history:
                    logger.info(f"   Including {len(history)} messages from history")
            
            if self.kafka_producer:
                await self._publish_prompt_request(
                    conversation_id=conversation_id,
                    conversation=conversation,
                    dialogue=dialogue,
                    prompt=prompt,
                    file_ids=file_ids,
                    history=history
                )
            else:
                logger.warning(f"⚠️ Kafka producer not available, event not published for dialogue: {dialogue.id}")
            
            return dialogue.id
            
        except ValueError:
            logger.error(f"❌ Invalid UUID format for conversation_id: {conversation_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for conversation_id")
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error creating dialogue: {str(e)}")
            logger.exception("Dialogue creation error details:")
            raise HTTPException(status_code=500, detail=f"Error creating dialogue: {str(e)}")
    
    async def _publish_prompt_request(
        self, 
        conversation_id: str, 
        conversation: Conversation, 
        dialogue: Dialogue, 
        prompt: str,
        file_ids: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ):
        """Publish prompt request event to Kafka
        
        Args:
            conversation_id: ID of the conversation
            conversation: Conversation object
            dialogue: Dialogue object
            prompt: User's prompt
            file_ids: Optional list of document IDs to restrict search context
            history: Optional conversation history
        """
        try:
            prompt_event = PromptAnswerRequestedEvent(
                conversation_id=conversation_id,
                dialogue_id=str(dialogue.id),
                prompt=prompt,
                user_id=str(conversation.user_id),
                file_ids=file_ids,
                history=history  
            )
            
            logger.info(f"📤 Publishing Kafka event for dialogue: {dialogue.id}")
            logger.debug(f"   Event ID: {prompt_event.event_id}")
            logger.debug(f"   Topic: {prompt_event.topic}")
            if file_ids:
                logger.info(f"   File IDs: {file_ids}")
            if history:
                logger.info(f"   History messages: {len(history)}")
            
            result = self.kafka_producer.produce(event=prompt_event, key=conversation_id)
            
            logger.info(f"✅ Kafka event published successfully for dialogue: {dialogue.id}")
            logger.info(f"   Event ID: {prompt_event.event_id}")
            if result:
                logger.info(f"   Topic: {result.get('topic')}, Partition: {result.get('partition')}, Offset: {result.get('offset')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish Kafka event for dialogue {dialogue.id}: {str(e)}")
            logger.exception("Kafka publishing error details:")
    
    def get_conversation(self, session: Session, conversation_id: str):
        """Get a conversation by ID with all dialogues"""
        try:
            conversation = session.get(Conversation, uuid.UUID(conversation_id))
            if not conversation:
                logger.warning(f"⚠️ Conversation not found: {conversation_id}")
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            dialogues = session.exec(
                select(Dialogue)
                .where(Dialogue.conversation_id == uuid.UUID(conversation_id))
                .order_by(Dialogue.created_at)
            ).all()
            
            logger.info(f"📖 Retrieved conversation {conversation_id} with {len(dialogues)} dialogues")
            
            return {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "dialogues": [
                    {
                        "id": d.id,
                        "prompt": d.prompt,
                        "answer": d.answer,
                        "created_at": d.created_at,
                        "updated_at": d.updated_at
                    } for d in dialogues
                ]
            }
        except ValueError:
            logger.error(f"❌ Invalid UUID format for conversation_id: {conversation_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for conversation_id")
    
    def get_user_conversations(self, session: Session, user_id: str):
        """Get all conversations for a user"""
        try:
            user_uuid = uuid.UUID(user_id)
            conversations = session.exec(
                select(Conversation)
                .where(Conversation.user_id == user_uuid)
                .order_by(Conversation.created_at.desc())
            ).all()
            
            logger.info(f"📋 Retrieved {len(conversations)} conversations for user: {user_id}")
            
            return [
                {
                    "id": c.id,
                    "user_id": c.user_id,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                    "title": f"Conversation {str(c.id)[:8]}",
                    "dialogue_count": len(session.exec(
                        select(Dialogue).where(Dialogue.conversation_id == c.id)
                    ).all())
                } for c in conversations
            ]
        except ValueError:
            logger.error(f"❌ Invalid UUID format for user_id: {user_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for user_id")
    
    def update_dialogue_answer(self, session: Session, dialogue_id: str, answer: str):
        """Update a dialogue with the assistant's answer"""
        try:
            dialogue = session.get(Dialogue, uuid.UUID(dialogue_id))
            if not dialogue:
                logger.warning(f"⚠️ Dialogue not found: {dialogue_id}")
                raise HTTPException(status_code=404, detail="Dialogue not found")
            
            dialogue.answer = answer
            dialogue.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(dialogue)
            
            logger.info(f"✅ Dialogue {dialogue_id} updated with answer (length: {len(answer)} chars)")
            
            return dialogue
        except ValueError:
            logger.error(f"❌ Invalid UUID format for dialogue_id: {dialogue_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for dialogue_id")
    
    def delete_conversation(self, session: Session, conversation_id: str):
        """Delete a conversation and all its dialogues"""
        try:
            conversation = session.get(Conversation, uuid.UUID(conversation_id))
            if not conversation:
                logger.warning(f"⚠️ Conversation not found for deletion: {conversation_id}")
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            session.delete(conversation)
            session.commit()
            
            logger.info(f"🗑️ Conversation deleted: {conversation_id}")
            
            return {"message": "Conversation deleted successfully"}
        except ValueError:
            logger.error(f"❌ Invalid UUID format for conversation_id: {conversation_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for conversation_id")
    
    def get_conversation_dialogues(
        self, 
        session: Session, 
        conversation_id: str, 
        limit: int = 50, 
        offset: int = 0
    ):
        """Get dialogues for a conversation with pagination"""
        try:
            conversation = session.get(Conversation, uuid.UUID(conversation_id))
            if not conversation:
                logger.warning(f"⚠️ Conversation not found: {conversation_id}")
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            dialogues = session.exec(
                select(Dialogue)
                .where(Dialogue.conversation_id == uuid.UUID(conversation_id))
                .order_by(Dialogue.created_at.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            
            logger.info(f"📖 Retrieved {len(dialogues)} dialogues for conversation {conversation_id}")
            
            return [
                {
                    "id": d.id,
                    "prompt": d.prompt,
                    "answer": d.answer,
                    "created_at": d.created_at,
                    "updated_at": d.updated_at
                } for d in dialogues
            ]
        except ValueError:
            logger.error(f"❌ Invalid UUID format for conversation_id: {conversation_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for conversation_id")
    
    def get_dialogue_by_id(self, session: Session, dialogue_id: str):
        """Get a single dialogue by ID"""
        try:
            dialogue = session.get(Dialogue, uuid.UUID(dialogue_id))
            if not dialogue:
                logger.warning(f"⚠️ Dialogue not found: {dialogue_id}")
                raise HTTPException(status_code=404, detail="Dialogue not found")
            
            logger.info(f"📖 Retrieved dialogue: {dialogue_id}")
            
            return {
                "id": dialogue.id,
                "conversation_id": dialogue.conversation_id,
                "prompt": dialogue.prompt,
                "answer": dialogue.answer,
                "created_at": dialogue.created_at,
                "updated_at": dialogue.updated_at
            }
        except ValueError:
            logger.error(f"❌ Invalid UUID format for dialogue_id: {dialogue_id}")
            raise HTTPException(status_code=400, detail="Invalid UUID format for dialogue_id")