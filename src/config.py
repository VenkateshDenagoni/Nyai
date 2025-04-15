import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base application settings
class BaseConfig:
    """Base configuration class."""
    # Application settings
    APP_NAME = "NYAI Legal Assistant"
    VERSION = "1.0.0"
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///nyai.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
        "pool_pre_ping": True
    }
    
    # Redis settings for caching and rate limiting
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS = 100
    
    # Cache settings
    CACHE_TYPE = "redis"
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 3600  # 1 hour
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_WINDOW = 60  # seconds
    MAX_REQUESTS_PER_WINDOW = 100  # Increased for production
    
    # Session settings
    SESSION_TYPE = "redis"
    SESSION_REDIS = REDIS_URL
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # Load balancing
    WORKER_COUNT = 4
    THREADS_PER_WORKER = 2
    
    # Log settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    LOG_FILE = "nyai_api.log"
    
    # API settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL = "gemini-2.0-flash-001"
    
    # Token and request limits
    MAX_PROMPT_LENGTH = 4000
    MAX_HISTORY_LENGTH = 10
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 1
    RETRY_MIN_SECONDS = 4
    RETRY_MAX_SECONDS = 10
    
    # Security settings
    AUTH_REQUIRED = False
    
    # Prompt template paths
    PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompts"
    
    # Feature flags
    ENABLE_CONTENT_SAFETY = True
    
    # Health checks
    HEALTH_CHECK_TIMEOUTS = 5  # seconds

class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    LOG_LEVEL = logging.DEBUG
    RATE_LIMIT_ENABLED = False
    
class ProductionConfig(BaseConfig):
    """Production configuration."""
    LOG_LEVEL = logging.WARNING
    AUTH_REQUIRED = True
    
    # Production database settings
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 50,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 1800,
        "pool_pre_ping": True
    }
    
    # Production Redis settings
    REDIS_URL = os.getenv("REDIS_URL")
    REDIS_MAX_CONNECTIONS = 200
    
    # Production worker settings
    WORKER_COUNT = 8
    THREADS_PER_WORKER = 4
    
    # Production rate limiting
    MAX_REQUESTS_PER_WINDOW = 200  # Higher limit for production
    
    # ... rest of existing config ...

class TestingConfig(BaseConfig):
    """Testing configuration."""
    LOG_LEVEL = logging.DEBUG
    CACHE_TTL = 1  # Short cache for testing
    ENABLE_CONTENT_SAFETY = False

# Select config based on environment
def get_config():
    """Return the appropriate configuration based on environment."""
    env = os.getenv("NYAI_ENV", "development").lower()
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    return configs.get(env, DevelopmentConfig)

# Export the active configuration
config = get_config() 