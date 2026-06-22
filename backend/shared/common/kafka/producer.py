import json
import logging
from typing import Optional, Dict, Any
from confluent_kafka import Producer as ConfluentProducer
from common.events.base_event import BaseEvent

logger = logging.getLogger(__name__)


class KafkaProducer:
    """Universal Kafka producer using confluent-kafka (official client)"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[ConfluentProducer] = None
    
    def start(self) -> None:
        """Start the Kafka producer"""
        try:
            conf = {
                'bootstrap.servers': self.bootstrap_servers,
                'acks': 'all',
                'retries': 3,
                'compression.type': 'gzip',
                'enable.idempotence': True,
            }
            self.producer = ConfluentProducer(conf)
            logger.info(f"✅ Kafka producer connected to {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"❌ Failed to start Kafka producer: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the Kafka producer"""
        if self.producer:
            self.producer.flush()
            logger.info("✅ Kafka producer stopped")
    
    def produce(self, event: BaseEvent, key: Optional[str] = None) -> Dict[str, Any]:
        """Produce event to Kafka using event's topic"""
        if not self.producer:
            raise RuntimeError("Kafka producer not started. Call start() first.")
        
        # Get topic - handle both instance property and class method
        if hasattr(event, 'topic'):
            if callable(event.topic):
                topic = event.topic()
            else:
                topic = event.topic
        else:
            topic = event.__class__.topic()
        
        value = json.dumps(event.model_dump(), default=str)
        
        # Delivery callback
        def delivery_report(err, msg):
            if err:
                logger.error(f"❌ Delivery failed: {err}")
            else:
                logger.info(
                    f"📤 Produced {event.__class__.__name__} to '{msg.topic()}' | "
                    f"Partition: {msg.partition()} | "
                    f"Offset: {msg.offset()}"
                )
        
        # Produce message (non-blocking)
        try:
            self.producer.produce(
                topic=topic,
                key=key,
                value=value.encode('utf-8'),
                callback=delivery_report
            )
            # Trigger delivery reports
            self.producer.poll(0)
            
            return {
                'topic': topic,
                'key': key,
                'event_id': event.event_id
            }
        except Exception as e:
            logger.error(f"❌ Failed to produce message: {e}")
            raise
    
    def flush(self) -> None:
        """Flush all pending messages"""
        if self.producer:
            self.producer.flush()
    
    def is_connected(self) -> bool:
        """Check if producer is connected"""
        return self.producer is not None


# Singleton instance
_producer: Optional[KafkaProducer] = None


def get_producer(bootstrap_servers: Optional[str] = None) -> KafkaProducer:
    """Get or create Kafka producer singleton"""
    global _producer
    if _producer is None:
        _producer = KafkaProducer(bootstrap_servers=bootstrap_servers or "localhost:9092")
        _producer.start()
    return _producer


def shutdown_producer() -> None:
    """Shutdown the Kafka producer"""
    global _producer
    if _producer:
        _producer.stop()
        _producer = None