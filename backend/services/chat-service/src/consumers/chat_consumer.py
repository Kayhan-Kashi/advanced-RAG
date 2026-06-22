import asyncio
import json
import logging
from typing import Optional
from confluent_kafka import Consumer
from api.websocket.manager import connection_manager
from services.conversation_service import ConversationService
from database.sqlite_session import get_session

logger = logging.getLogger(__name__)


class ChatConsumer:
    """Dedicated Kafka consumer for chat completion events"""
    
    def __init__(self, bootstrap_servers: str, group_id: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topic = topic
        self.consumer: Optional[Consumer] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    def start(self):
        """Start the consumer"""
        conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 30000,
            'max.poll.interval.ms': 300000,
        }
        
        self.consumer = Consumer(conf)
        self.consumer.subscribe([self.topic])
        self.running = True
        
        logger.info(f"✅ ChatConsumer started")
        logger.info(f"   Topic: {self.topic}")
        logger.info(f"   Group ID: {self.group_id}")
        logger.info(f"   Bootstrap: {self.bootstrap_servers}")
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("ChatConsumer stopped")
    
    async def consume(self):
        """Background task to consume messages"""
        if not self.consumer:
            raise RuntimeError("ChatConsumer not started")
        
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Poll for messages (non-blocking)
                msg = await loop.run_in_executor(None, self.consumer.poll, 1.0)
                
                if msg is None:
                    await asyncio.sleep(0.1)
                    continue
                
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                # Process message
                try:
                    message = json.loads(msg.value().decode('utf-8'))
                    await self._process_message(message)
                    logger.debug(f"Message processed at offset {msg.offset()}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: dict):
        """Process incoming Kafka message"""
        try:
            event_type = message.get("event_type")
            
            if event_type == "prompt_answer_completed":
                await self._handle_completion_event(message)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_completion_event(self, message: dict):
        """Handle prompt_answer_completed event"""
        conversation_id = message.get("conversation_id")
        dialogue_id = message.get("dialogue_id")
        full_answer = message.get("full_answer")
        
        logger.info(f"📥 Received completion event")
        logger.info(f"   Conversation: {conversation_id[:8]}...")
        logger.info(f"   Dialogue: {dialogue_id[:8]}...")
        logger.info(f"   Answer length: {len(full_answer)} characters")
        
        # Update database
        try:
            with next(get_session()) as db:
                service = ConversationService()
                service.update_dialogue_answer(
                    session=db,
                    dialogue_id=dialogue_id,
                    answer=full_answer
                )
            logger.info(f"✅ Dialogue updated in database")
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
        
        success = await connection_manager.send_answer(
            conversation_id=conversation_id,
            answer=full_answer  
        )
        
        if success:
            logger.info(f"✅ Answer sent via WebSocket")
        else:
            logger.warning(f"⚠️ WebSocket not available for conversation {conversation_id[:8]}...")