from datetime import datetime, timedelta
import hashlib
import json
from typing import Optional, Any
from sqlalchemy.exc import SQLAlchemyError

from src.app import db, cache, redis_available
from src.models import ResponseCache
from src.config import config
from src.utils.logger import logger

class CacheService:
    """Service to handle response caching."""
    
    @staticmethod
    def get_cached_response(prompt: str, context: Optional[str] = None) -> Optional[str]:
        """
        Get a cached response.
        
        Args:
            prompt: The user prompt
            context: Optional additional context
            
        Returns:
            The cached response or None if not found
        """
        cache_key = CacheService._generate_cache_key(prompt, context)
        
        # Try to get from in-memory cache first
        cached = cache.get(cache_key)
        if cached:
            return cached
            
        # If Redis not available, try database
        if not redis_available:
            try:
                cached_item = ResponseCache.query.filter_by(cache_key=cache_key).first()
                if cached_item and cached_item.expires_at > datetime.utcnow():
                    # Store in memory cache for faster access next time
                    cache.set(cache_key, cached_item.response)
                    return cached_item.response
            except SQLAlchemyError as e:
                logger.error(f"Database error getting cached response: {e}")
                
        return None
    
    @staticmethod
    def cache_response(prompt: str, response: str, context: Optional[str] = None) -> None:
        """
        Cache a response.
        
        Args:
            prompt: The user prompt
            response: The AI response
            context: Optional additional context
        """
        cache_key = CacheService._generate_cache_key(prompt, context)
        ttl = config.CACHE_DEFAULT_TIMEOUT
        
        # Cache in memory/Redis
        cache.set(cache_key, response, timeout=ttl)
        
        # If Redis not available, also cache in database
        if not redis_available:
            try:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                # Check if already exists
                cached_item = ResponseCache.query.filter_by(cache_key=cache_key).first()
                
                if cached_item:
                    # Update existing
                    cached_item.response = response
                    cached_item.expires_at = expires_at
                else:
                    # Create new
                    cached_item = ResponseCache(
                        cache_key=cache_key,
                        response=response,
                        expires_at=expires_at
                    )
                    db.session.add(cached_item)
                    
                db.session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database error caching response: {e}")
                db.session.rollback()
    
    @staticmethod
    def invalidate_cache(prompt: str, context: Optional[str] = None) -> None:
        """
        Invalidate a cached response.
        
        Args:
            prompt: The user prompt
            context: Optional additional context
        """
        cache_key = CacheService._generate_cache_key(prompt, context)
        
        # Remove from memory/Redis cache
        cache.delete(cache_key)
        
        # If Redis not available, also remove from database
        if not redis_available:
            try:
                ResponseCache.query.filter_by(cache_key=cache_key).delete()
                db.session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database error invalidating cache: {e}")
                db.session.rollback()
    
    @staticmethod
    def _generate_cache_key(prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a cache key for a prompt and context.
        
        Args:
            prompt: The user prompt
            context: Optional additional context
            
        Returns:
            A hash string to use as cache key
        """
        key_content = f"{prompt}:{context or ''}"
        return hashlib.md5(key_content.encode()).hexdigest() 