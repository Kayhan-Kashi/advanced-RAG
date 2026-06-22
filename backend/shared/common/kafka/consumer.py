import json
import logging
import asyncio
from typing import List, Dict, Any
from confluent_kafka import Consumer as ConfluentConsumer
from shared.common.message_bus.interfaces import MessageBus

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """
    Generic Kafka consumer that uses MessageBus for routing.
    Receives the bus instance at start time.
    """
    
    def __init__(self, bootstrap_servers: str, group_id: str, topics: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.consumer = None
        self.bus = None
        self.running = False
    
    def start(self, message_bus: MessageBus) -> None:
        """Start consumer with message bus for routing"""
        self.bus = message_bus
        
        conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        }
        
        self.consumer = ConfluentConsumer(conf)
        self.consumer.subscribe(self.topics)
        
        logger.info(f"✅ Consumer started on topics: {self.topics}")
    
    def _route_message(self, message: Dict[str, Any]):
        """Route message to appropriate handler using message bus"""
        event_type = message.get('event_type')
        
        if not event_type:
            logger.warning(f"Message missing event_type")
            return
        
        handler = self.bus.get_handler(event_type)
        
        if not handler:
            logger.warning(f"No handler registered for: {event_type}")
            return
        
        try:
            asyncio.run(handler.handle(None, message))
        except Exception as e:
            logger.error(f"Handler error for {event_type}: {e}")
    
    def consume_loop(self) -> None:
        """Main consume loop"""
        self.running = True
        
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                try:
                    message = json.loads(msg.value().decode('utf-8'))
                    self._route_message(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop consumer"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer stopped")