# llm-service/src/workers/worker.py
import asyncio
import json
import sys
import traceback
import os
from confluent_kafka import Consumer, Producer, KafkaError  
from injector import Injector
import logging

from src.dependencies import injector 
from common.message_bus.bus import MessageBus #type: ignore
from src.registry import register_handlers, get_consumption_topics, get_event_types
from common.events.document_uploaded import DocumentUploadedEvent #type: ignore
from common.events.prompt_answer_requested import PromptAnswerRequestedEvent #type: ignore
from src.handlers.document_uploaded_handler import DocumentUploadedHandler
from src.handlers.prompt_answer_requested_handler import PromptAnswerRequestedHandler



# ============================================
# Kafka settings
# ============================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
CONSUMER_GROUP = os.getenv('KAFKA_CONSUMER_GROUP', 'llm-worker-group')


# ============================================
# Get event types from registry
# ============================================

EVENT_TYPES = get_event_types()


# ============================================
# Dependency Injection
# ============================================

bus = MessageBus(injector)


# Register all handlers using the registry
register_handlers(bus)

# Get topics to consume from registry and convert to strings
SUBSCRIBE_TOPICS_RAW = get_consumption_topics(bus)
# Convert to strings if they are properties
SUBSCRIBE_TOPICS = [str(topic) if not isinstance(topic, str) else topic for topic in SUBSCRIBE_TOPICS_RAW]


# ============================================
# Helper: publish events to Kafka
# ============================================

producer: Producer = None


def get_producer() -> Producer:
    """Get or create Kafka producer"""
    global producer
    if producer is None:
        conf = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'acks': 'all',
            'retries': 3,
            'compression.type': 'gzip',
        }
        producer = Producer(conf)
        print(f"[llm-service] ✅ Kafka producer connected to {KAFKA_BOOTSTRAP_SERVERS}", flush=True)
    return producer


def make_delivery_report(event_type, topic):
    """Factory function to create delivery report with captured values"""
    def delivery_report(err, msg):
        if err:
            print(f"[llm-service] ❌ Failed to publish {event_type}: {err}", flush=True)
        else:
            print(f"[llm-service] 📤 Published {event_type} to {topic}", flush=True)
    return delivery_report


async def publish_event(event):
    """Publish event to Kafka"""
    try:
        prod = get_producer()
        
        event_data = json.dumps(event.model_dump(), default=str).encode('utf-8')
        
        # Get event_type as string
        if hasattr(event, 'event_type'):
            if callable(event.event_type):
                event_type = event.event_type()
            else:
                event_type = event.event_type
        else:
            event_type = event.__class__.__name__
        
        # Get topic as string
        if hasattr(event, 'topic'):
            if callable(event.topic):
                topic = event.topic()
            else:
                topic = event.topic
        else:
            topic = event.__class__.topic()
        
        key = getattr(event, 'conversation_id', None) or getattr(event, 'document_id', None)
        
        prod.produce(
            topic=topic,
            key=key,
            value=event_data,
            callback=make_delivery_report(event_type, topic)
        )
        
        # ✅ Flush to ensure callback executes before next event
        prod.flush()
        
    except Exception as exc:
        print(f"[llm-service] ❌ Error publishing event: {exc}", flush=True)
        traceback.print_exc()
        

# ============================================
# Core message processor
# ============================================

async def handle_message(message: dict):
    """Process a single Kafka message"""
    try:
        event_type = message.get("event_type")
        
        print(f"[llm-service] 📩 Received {event_type}", flush=True)
        
        event_class = EVENT_TYPES.get(event_type)
        if not event_class:
            print(f"[llm-service] ⚠ Unknown event: {event_type}", flush=True)
            return
        
        event = event_class(**message)
        
        result = await bus.dispatch(event, db=None)
        
        if result:
            await publish_event(result)
            
    except Exception as exc:
        print(f"[llm-service] ❌ Error processing message: {exc}", flush=True)
        traceback.print_exc()


# ============================================
# Kafka consumer setup
# ============================================

async def create_consumer() -> Consumer:
    """Create and configure Kafka consumer"""
    
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': CONSUMER_GROUP,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'auto.commit.interval.ms': 5000,
        'allow.auto.create.topics': True,  
    }
    
    consumer = Consumer(conf)
    
    if SUBSCRIBE_TOPICS:
        consumer.subscribe(SUBSCRIBE_TOPICS)
        print(f"[llm-service] ✅ Consumer subscribed to: {', '.join(SUBSCRIBE_TOPICS)}", flush=True)
    else:
        print(f"[llm-service] ⚠ No topics to subscribe to", flush=True)
    
    print(f"[llm-service]    Group ID: {CONSUMER_GROUP}", flush=True)
    
    return consumer


# ============================================
# Main worker
# ============================================

async def main():
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    print("[llm-service] 🚀 Starting worker...", flush=True)
    
    consumer = await create_consumer()
    
    print(f"[llm-service] ✅ Worker listening to: {list(EVENT_TYPES.keys())}", flush=True)
    print("[llm-service] ⏳ Waiting for events...", flush=True)
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._UNKNOWN_TOPIC_OR_PART:
                    print(f"[llm-service] ⚠️ Topic not available yet, waiting 3s...", flush=True)
                    await asyncio.sleep(3)
                    continue
                else:
                    print(f"[llm-service] Consumer error: {msg.error()}", flush=True)
                    continue
            
            try:
                message = json.loads(msg.value().decode('utf-8'))
                await handle_message(message)
                
            except json.JSONDecodeError as e:
                print(f"[llm-service] ❌ Failed to decode message: {e}", flush=True)
                continue
    except KeyboardInterrupt:
        print("[llm-service] ⚠ Worker interrupted", flush=True)
    except Exception as e:
        print(f"[llm-service] ❌ Unexpected error: {e}", flush=True)
        traceback.print_exc()
    finally:
        consumer.close()
        if producer:
            producer.flush()
        print("[llm-service] ✅ Worker shutdown", flush=True)


if __name__ == "__main__":
    asyncio.run(main())