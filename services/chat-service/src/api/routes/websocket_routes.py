# elearning-service/src/api/routes/websocket_routes.py
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi_injector import Injected
from sqlmodel import Session

from api.websocket.manager import connection_manager
from database.sqlite_session import get_session
from services.conversation_service import ConversationService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    user_id: str,
    db: Session = Depends(get_session),
    service: ConversationService = Injected(ConversationService)
):
    """WebSocket endpoint for real-time communication"""
    
    logger.info(f"🔌 WebSocket connection attempt from user: {user_id}")
    
    try:
        await websocket.accept()
        logger.info(f"✅ WebSocket accepted for user: {user_id}")
        
        await connection_manager.connect(websocket, user_id)
        logger.info(f"📝 User {user_id} registered in connection manager")
        
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                
                logger.info(f"📨 Received message from user {user_id}: {msg_type}")
                logger.debug(f"   Message data: {data}")
                
                if msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.debug(f"🏓 Sent pong to user {user_id}")
                
                elif msg_type == "chat":
                    conversation_id = data.get("conversation_id")
                    prompt = data.get("prompt")
                    file_ids: Optional[List[str]] = data.get("file_ids", [])  # ← Extract file_ids
                    
                    logger.info(f"💬 Chat message from user {user_id} in conversation {conversation_id}")
                    logger.info(f"   Prompt: {prompt[:100]}...")
                    logger.info(f"   File IDs: {file_ids}")
                    
                    connection_manager.register_conversation(conversation_id, user_id)
                    
                    dialogue_id = await service.create_dialogue(
                        session=db,
                        conversation_id=conversation_id,
                        prompt=prompt,
                        answer=None,
                        file_ids=file_ids  
                    )
                    
                    logger.info(f"📝 Dialogue created: {dialogue_id}")
                    logger.info(f"   Associated file IDs: {file_ids}")
                    logger.info(f"📤 Kafka event published by ConversationService")
                    
                    await websocket.send_json({
                        "type": "ack",
                        "conversation_id": conversation_id,
                        "dialogue_id": str(dialogue_id),
                        "file_ids": file_ids,  # ← Include file_ids in acknowledgment
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.debug(f"✅ Sent acknowledgment to user {user_id}")
                    
                
                else:
                    logger.warning(f"⚠️ Unknown message type from user {user_id}: {msg_type}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except WebSocketDisconnect:
                logger.info(f"❌ User {user_id} disconnected")
                break
                
            except Exception as e:
                logger.error(f"❌ Error processing message from user {user_id}: {e}")
                logger.exception("Stack trace:")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except:
                    pass
                
    except Exception as e:
        logger.error(f"❌ WebSocket error for user {user_id}: {e}")
        logger.exception("Stack trace:")
        await websocket.close()