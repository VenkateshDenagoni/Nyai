from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from src.app import db

class Conversation(db.Model):
    """Model for storing conversation history."""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(36), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with messages
    messages = relationship('Message', back_populates='conversation', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('ix_conversations_session_id_user_id', 'session_id', 'user_id'),
    )

class Message(db.Model):
    """Model for storing individual messages in conversations."""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), nullable=False)
    role = Column(String(10), nullable=False)  # 'user' or 'ai'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with conversation
    conversation = relationship('Conversation', back_populates='messages')
    
    # Indexes
    __table_args__ = (
        Index('ix_messages_conversation_id_created_at', 'conversation_id', 'created_at'),
    )

class ResponseCache(db.Model):
    """Model for caching LLM responses."""
    __tablename__ = 'response_cache'
    
    id = Column(Integer, primary_key=True)
    cache_key = Column(String(64), unique=True, nullable=False, index=True)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('ix_response_cache_cache_key_expires_at', 'cache_key', 'expires_at'),
    )

class RateLimit(db.Model):
    """Model for tracking rate limits."""
    __tablename__ = 'rate_limits'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(128), nullable=False, index=True)
    count = Column(Integer, default=0)
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('ix_rate_limits_client_id_window', 'client_id', 'window_start', 'window_end'),
    ) 