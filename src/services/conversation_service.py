from datetime import datetime
from uuid import uuid4
from typing import List, Dict, Optional, Any
from sqlalchemy.exc import SQLAlchemyError

from src.app import db, cache
from src.models import Conversation, Message
from src.config import config
from src.utils.logger import logger

class ConversationService:
    """Service to handle conversation storage and retrieval."""
    
    @staticmethod
    def get_conversation(session_id: str) -> List[Dict[str, str]]:
        """
        Get a conversation history from the database.
        
        Args:
            session_id: The session ID to get history for
            
        Returns:
            A list of message dictionaries
        """
        try:
            # Try to get from cache first
            cache_key = f"conversation:{session_id}"
            cached = cache.get(cache_key)
            if cached:
                return cached
                
            # Get from database
            conversation = Conversation.query.filter_by(session_id=session_id).first()
            
            if conversation:
                # Convert to the expected format
                history = []
                messages = Message.query.filter_by(
                    conversation_id=conversation.id
                ).order_by(Message.created_at).all()
                
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        history.append({
                            "user": messages[i].content,
                            "ai": messages[i+1].content
                        })
                        
                # Cache the results
                cache.set(cache_key, history)
                return history
            else:
                return []
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting conversation: {e}")
            # Fallback to empty history in case of database error
            return []
    
    @staticmethod
    def save_conversation(session_id: str, user_message: str, ai_response: str, user_id: Optional[str] = None) -> None:
        """
        Save a conversation exchange to the database.
        
        Args:
            session_id: The session ID
            user_message: The user's message
            ai_response: The AI's response
            user_id: Optional user ID
        """
        try:
            # Find or create conversation
            conversation = Conversation.query.filter_by(session_id=session_id).first()
            
            if not conversation:
                conversation = Conversation(
                    session_id=session_id,
                    user_id=user_id
                )
                db.session.add(conversation)
                db.session.flush()  # Generate ID before creating messages
                
            # Add user message
            user_msg = Message(
                conversation_id=conversation.id,
                role="user",
                content=user_message
            )
            db.session.add(user_msg)
            
            # Add AI response
            ai_msg = Message(
                conversation_id=conversation.id,
                role="ai",
                content=ai_response
            )
            db.session.add(ai_msg)
            
            # Update conversation timestamp
            conversation.updated_at = datetime.utcnow()
            
            # Commit changes
            db.session.commit()
            
            # Invalidate cache
            cache_key = f"conversation:{session_id}"
            cache.delete(cache_key)
            
        except SQLAlchemyError as e:
            logger.error(f"Database error saving conversation: {e}")
            db.session.rollback()
    
    @staticmethod
    def clear_conversation(session_id: str) -> bool:
        """
        Clear a conversation history.
        
        Args:
            session_id: The session ID to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conversation = Conversation.query.filter_by(session_id=session_id).first()
            
            if conversation:
                # Delete all messages
                Message.query.filter_by(conversation_id=conversation.id).delete()
                
                # Delete conversation
                db.session.delete(conversation)
                db.session.commit()
                
                # Invalidate cache
                cache_key = f"conversation:{session_id}"
                cache.delete(cache_key)
                
                return True
            else:
                return False
                
        except SQLAlchemyError as e:
            logger.error(f"Database error clearing conversation: {e}")
            db.session.rollback()
            return False
            
    @staticmethod
    def generate_session_id() -> str:
        """Generate a new session ID."""
        return str(uuid4()) 