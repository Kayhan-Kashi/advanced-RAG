import logging
from typing import Optional, Any
from injector import inject
from common.events import PromptAnswerCompletedEvent #type: ignore
from services.conversation_service import ConversationService

logger = logging.getLogger(__name__)


class PromptAnswerCompletedHandler:
    """Handler for PromptAnswerCompletedEvent - updates dialogue in database"""
    
    @inject
    def __init__(self, conversation_service: ConversationService):
        """
        Initialize handler with injected ConversationService
        
        Args:
            conversation_service: Injected ConversationService
        """
        self.conversation_service = conversation_service
    
    async def handle(self, event: PromptAnswerCompletedEvent, db: Optional[Any] = None):
        """
        Handle completed answer: update database only
        
        Args:
            event: The completed event containing the answer
            db: Database session (injected by message bus)
        """
        try:
            logger.info(f"📥 [Chat] Processing completed answer")
            logger.info(f"   Event ID: {event.event_id[:8]}...")
            logger.info(f"   Conversation ID: {event.conversation_id[:8]}...")
            logger.info(f"   Dialogue ID: {event.dialogue_id[:8]}...")
            logger.info(f"   Answer length: {len(event.full_answer)} characters")
            
            self.conversation_service.update_dialogue_answer(
                session=db,
                dialogue_id=event.dialogue_id,
                answer=event.full_answer
            )
            logger.info(f"✅ [Chat] Dialogue updated in database")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [Chat] Error: {e}")
            logger.exception("Stack trace:")
            raise