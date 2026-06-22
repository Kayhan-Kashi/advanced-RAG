from datetime import datetime
import logging
from typing import Dict
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_routing: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        self.active_connections[user_id] = websocket
        logger.info(f"✅ User {user_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            conversations_to_remove = [
                conv_id for conv_id, uid in self.conversation_routing.items() if uid == user_id
            ]
            for conv_id in conversations_to_remove:
                del self.conversation_routing[conv_id]
            logger.info(f"❌ User {user_id} disconnected. Total connections: {len(self.active_connections)}")
    
    def register_conversation(self, conversation_id: str, user_id: str):
        """Register a conversation to a user for WebSocket routing"""
        self.conversation_routing[conversation_id] = user_id
        logger.info(f"📝 Conversation {conversation_id[:8]}... registered to user {user_id}")
    
    async def send_answer(self, conversation_id: str, answer: str) -> bool:
        """Send answer to user based on conversation routing"""
        user_id = self.conversation_routing.get(conversation_id)
        
        if not user_id:
            logger.warning(f"❌ No user found for conversation {conversation_id[:8]}...")
            return False
        
        if user_id not in self.active_connections:
            logger.warning(f"❌ User {user_id} not connected")
            return False
        
        try:
            await self.active_connections[user_id].send_json({
                "type": "answer",
                "conversation_id": conversation_id,
                "answer": answer, 
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"📤 Answer sent to user {user_id} for conversation {conversation_id[:8]}...")
            logger.debug(f"   Answer length: {len(answer)} characters")
            return True
        except Exception as e:
            logger.error(f"Failed to send answer to user {user_id}: {e}")
            return False
    
    async def send_error(self, conversation_id: str, error: str) -> bool:
        """Send error to user based on conversation routing"""
        user_id = self.conversation_routing.get(conversation_id)
        
        if not user_id:
            logger.warning(f"❌ No user found for conversation {conversation_id[:8]}...")
            return False
        
        if user_id not in self.active_connections:
            logger.warning(f"❌ User {user_id} not connected")
            return False
        
        try:
            await self.active_connections[user_id].send_json({
                "type": "error",
                "conversation_id": conversation_id,
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"⚠️ Error sent to user {user_id} for conversation {conversation_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send error to user {user_id}: {e}")
            return False


connection_manager = ConnectionManager()