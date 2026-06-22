import os

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from injector import Injector
from fastapi_injector import InjectorMiddleware, attach_injector

from common.kafka.producer import get_producer, shutdown_producer #type: ignore
from database.sqlite_session import create_db_and_tables
from api.routes.conversation_routes import router as conversation_router
from api.routes.websocket_routes import router as websocket_router
from api.routes.document_routes import router as document_router
from consumers.chat_consumer import ChatConsumer
from dependencies import DependencyInjection
import asyncio

chat_consumer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_consumer
    
    print("🚀 Starting E-Learning Service...")
    
    # Initialize database
    create_db_and_tables()
    print("✅ Database ready")
    
    # Get Kafka bootstrap servers from environment variable
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    print(f"📡 Kafka bootstrap servers: {bootstrap_servers}")
    
    get_producer(bootstrap_servers=bootstrap_servers)
    print("✅ Kafka producer ready")
    
    # Initialize and start ChatConsumer
    chat_consumer = ChatConsumer(
        bootstrap_servers=bootstrap_servers,
        group_id="elearning-chat-group",
        topic="prompt-answer-completed"
    )
    chat_consumer.start()
    
    # Start consumer background task
    consumer_task = asyncio.create_task(chat_consumer.consume())
    app.state.consumer_task = consumer_task
    
    print("✅ ChatConsumer started")
    print("✅ WebSocket manager ready")
    print("✅ Application started successfully")
    
    yield
    
    print("🛑 Shutting down...")
    if chat_consumer:
        chat_consumer.stop()
    shutdown_producer()
    if consumer_task:
        consumer_task.cancel()
    print("✅ Application shut down")


injector = Injector([DependencyInjection()])

app = FastAPI(
    title="E-Learning Service",
    description="API for managing conversations and dialogues",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(InjectorMiddleware, injector=injector)
attach_injector(app, injector)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(conversation_router)

app.include_router(websocket_router)

app.include_router(document_router)


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "consumer_running": chat_consumer.running if chat_consumer else False
    }


@app.options("/{rest_of_path:path}")
async def options_handler():
    return {"message": "OK"}


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)