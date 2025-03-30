import time
from functools import wraps
from flask import request, g, jsonify, current_app
import redis
from werkzeug.exceptions import TooManyRequests

from src.config import config
from src.utils.logger import logger, generate_request_id

# Setup Redis connection for rate limiting (will connect lazily)
rate_limit_redis = None
if config.RATE_LIMIT_ENABLED:
    try:
        import redis
        # Using redis to track rate limits (connection string would be in env vars in production)
        rate_limit_redis = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
    except (ImportError, redis.exceptions.ConnectionError):
        logger.warning("Redis not available. Rate limiting will use in-memory storage (not suitable for multiple instances).")
        rate_limit_store = {}

# Request tracking middleware
def request_middleware():
    """Middleware to track requests and add request ID."""
    def decorator(app):
        @app.before_request
        def before_request():
            # Generate and store request ID
            g.request_id = generate_request_id()
            g.start_time = time.time()
            
            # Log request details
            logger.info(f"Request {request.method} {request.path} from {request.remote_addr}")
            
        @app.after_request
        def after_request(response):
            # Calculate request duration
            duration = time.time() - g.get('start_time', time.time())
            duration_ms = round(duration * 1000, 2)
            
            # Add request ID and timing headers
            response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
            response.headers['X-Request-Duration-ms'] = str(duration_ms)
            
            # Log response details
            logger.info(f"Response {response.status_code} completed in {duration_ms}ms")
            
            return response
            
        @app.errorhandler(Exception)
        def handle_exception(e):
            # Log unhandled exceptions
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            
            # Return JSON response for API
            return jsonify({
                "error": "Internal server error",
                "request_id": g.get('request_id', 'unknown')
            }), 500
            
        return app
    
    return decorator

# Rate limiting decorator
def rate_limit(limit=None, window=None):
    """Rate limiting decorator for API endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not config.RATE_LIMIT_ENABLED:
                return f(*args, **kwargs)
                
            # Use config defaults if not specified
            rate_limit = limit or config.MAX_REQUESTS_PER_WINDOW
            time_window = window or config.RATE_LIMIT_WINDOW
            
            # Get client identifier (IP + route)
            client_id = f"{request.remote_addr}:{request.path}"
            
            # Check rate limit using Redis if available
            if rate_limit_redis:
                try:
                    current = rate_limit_redis.get(client_id)
                    if current is not None and int(current) >= rate_limit:
                        logger.warning(f"Rate limit exceeded for {client_id}")
                        response = jsonify({
                            "error": "Rate limit exceeded",
                            "retry_after": time_window
                        })
                        response.headers['Retry-After'] = time_window
                        return response, 429
                    
                    # Increment counter
                    pipe = rate_limit_redis.pipeline()
                    pipe.incr(client_id)
                    pipe.expire(client_id, time_window)
                    pipe.execute()
                except redis.exceptions.RedisError:
                    logger.error("Redis error during rate limiting")
            else:
                # Fallback to in-memory rate limiting
                now = time.time()
                key = f"{client_id}:{int(now / time_window)}"
                
                if key in rate_limit_store and rate_limit_store[key] >= rate_limit:
                    logger.warning(f"Rate limit exceeded for {client_id}")
                    response = jsonify({
                        "error": "Rate limit exceeded",
                        "retry_after": time_window
                    })
                    response.headers['Retry-After'] = time_window
                    return response, 429
                
                # Clean up old entries
                for k in list(rate_limit_store.keys()):
                    if int(now / time_window) > int(k.split(':')[-1]):
                        rate_limit_store.pop(k, None)
                
                # Increment counter
                rate_limit_store[key] = rate_limit_store.get(key, 0) + 1
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Authentication middleware
def auth_required(f):
    """Authentication middleware for protected endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not config.AUTH_REQUIRED:
            return f(*args, **kwargs)
            
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        
        # In a real app, you'd validate against a database or auth service
        # This is simplified for demo purposes
        if not api_key or api_key != current_app.config.get('API_KEY'):
            logger.warning(f"Unauthorized request from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
            
        return f(*args, **kwargs)
    return decorated 