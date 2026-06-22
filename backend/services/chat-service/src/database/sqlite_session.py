import os

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event, select
from contextlib import contextmanager
from typing import Generator
import uuid

# DATABASE_URL = "sqlite:///./database.db"
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./database.db')
# Fixed UUIDs for User1 and User2
USER1_ID = uuid.UUID('12345678-1234-5678-1234-567812345678')
USER2_ID = uuid.UUID('87654321-4321-8765-4321-876543210987')

engine = create_engine(
    DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False},
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

def create_db_and_tables():
    """Create database tables and initialize with seed data"""
    SQLModel.metadata.create_all(engine)
    print("✓ Database tables created successfully!")
    
    init_seed_data()

def init_seed_data():
    """Insert initial seed data (users, etc.)"""
    from database.models import User
    
    with Session(engine) as session:
        existing_user1 = session.exec(
            select(User).where(User.username == "user1")
        ).first()
        
        existing_user2 = session.exec(
            select(User).where(User.username == "user2")
        ).first()
        
        if not existing_user1:
            user1 = User(
                id=USER1_ID,
                name="User1", 
                username="user1", 
                password="123456"
            )
            session.add(user1)
            print(f"  ✓ Added user: {user1.name} (@{user1.username}) with ID: {user1.id}")
        else:
            print(f"  ⚠ User1 already exists, skipping")
        
        if not existing_user2:
            user2 = User(
                id=USER2_ID,
                name="User2", 
                username="user2", 
                password="123456"
            )
            session.add(user2)
            print(f"  ✓ Added user: {user2.name} (@{user2.username}) with ID: {user2.id}")
        else:
            print(f"  ⚠ User2 already exists, skipping")
        
        session.commit()
        print("✓ Seed data initialized!")

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

class DatabaseSessionManager:
    def __enter__(self):
        self.session = Session(engine)
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type:
                self.session.rollback()
                print(f"⚠️ Error occurred, rolled back: {exc_val}")
            else:
                self.session.commit()
        finally:
            self.session.close()