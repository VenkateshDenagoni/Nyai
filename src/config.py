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
    
    # Log settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    LOG_FILE = "nyai_api.log"
    
    # API settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL = "gemini-2.0-flash-001"
    
    # Cache settings
    CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)
    
    # Token and request limits
    MAX_PROMPT_LENGTH = 4000
    MAX_HISTORY_LENGTH = 10
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 1
    RETRY_MIN_SECONDS = 4
    RETRY_MAX_SECONDS = 10
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_WINDOW = 60  # seconds
    MAX_REQUESTS_PER_WINDOW = 10
    
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
    # In production, you would use a secrets manager instead of env vars
    # This is simplified for demo purposes
    
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