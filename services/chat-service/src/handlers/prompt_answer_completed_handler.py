import logging
from typing import Optional, Any
from injector import inject
from common.events import PromptAnswerCompletedEvent #type: ignore
from src.services.conversation_service import ConversationService
from api.websocket.manager import connection_manager

logger = logging.getLogger(__name__)


class PromptAnswerCompletedHandler:
    """Handler for PromptAnswerCompletedEvent - updates dialogue and sends WebSocket"""
    
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
        Handle completed answer: update database and send via WebSocket
        
        Args:
            event: The completed event containing the answer
            db: Database session (injected by message bus)
        """
        try:
            logger.info(f"📥 [E-Learning] Processing completed answer")
            logger.info(f"   Event ID: {event.event_id[:8]}...")
            logger.info(f"   Conversation ID: {event.conversation_id[:8]}...")
            logger.info(f"   Dialogue ID: {event.dialogue_id[:8]}...")
            logger.info(f"   Answer length: {len(event.full_answer)} characters")
            
            self.conversation_service.update_dialogue_answer(
                session=db,
                dialogue_id=event.dialogue_id,
                answer=event.full_answer
            )
            logger.info(f"✅ [E-Learning] Dialogue updated in database")
            
            success = await connection_manager.send_answer(
                conversation_id=event.conversation_id,
                full_answer=event.full_answer
            )
            
            if success:
                logger.info(f"✅ [E-Learning] Answer sent via WebSocket for conversation {event.conversation_id[:8]}...")
            else:
                logger.warning(f"⚠️ [E-Learning] WebSocket not available for conversation {event.conversation_id[:8]}...")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ [E-Learning] Error: {e}")
            logger.exception("Stack trace:")
            raise