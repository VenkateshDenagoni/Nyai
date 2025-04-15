from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import logging
from werkzeug.middleware.dispatcher import DispatcherMiddleware

try:
    from prometheus_client import make_wsgi_app
    prometheus_available = True
except ImportError:
    prometheus_available = False

from src.routes.ai_routes import ai_routes
from src.utils.errors import register_error_handlers
from src.middleware import request_middleware
from src.config import config
from src.utils.logger import logger

# Initialize extensions
db = SQLAlchemy()

# Check Redis availability
redis_available = True
try:
    redis_client = redis.Redis.from_url(
        config.REDIS_URL, 
        socket_connect_timeout=1
    )
    redis_client.ping()
except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
    redis_available = False
    logger.warning("Redis is not available. Using fallback in-memory storage.")

# Configure caching based on Redis availability
if redis_available:
    cache = Cache()
else:
    cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})

# Configure session based on Redis availability
session = Session()

# Configure rate limiter based on Redis availability
if redis_available:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per minute"],
        storage_uri=config.REDIS_URL
    )
else:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per minute"],
        storage_uri="memory://"
    )

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config)
    
    # Override Redis-dependent settings if Redis is unavailable
    if not redis_available:
        app.config['CACHE_TYPE'] = 'SimpleCache'
        app.config['SESSION_TYPE'] = 'filesystem'
    
    # Initialize extensions
    CORS(app)
    db.init_app(app)
    cache.init_app(app)
    session.init_app(app)
    limiter.init_app(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Apply middleware
    request_middleware()(app)
    
    # Register blueprints
    app.register_blueprint(ai_routes, url_prefix="/api")
    
    # Add health check endpoint
    @app.route("/health")
    def health_check():
        return {"status": "healthy"}
    
    # Add metrics endpoint if prometheus is available
    if prometheus_available:
        app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
            '/metrics': make_wsgi_app()
        })
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    app.run(port=5000, debug=True)
