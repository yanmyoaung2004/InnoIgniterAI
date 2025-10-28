from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.database import Base
import uuid


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    unique_id = Column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()) 
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False, default="New Chat")
    # summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    role = Column(String(50), nullable=False) 
    content = Column(Text, nullable=False)
    reason = Column(Text, nullable=True)
    time_stamp = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="messages")
    images = relationship("Image", back_populates="message", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    image_url = Column(String(255), nullable=False)
    time_stamp = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", back_populates="images")