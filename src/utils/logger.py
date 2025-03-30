import logging
import uuid
import sys
from logging.handlers import RotatingFileHandler
from functools import wraps
from flask import request, g, has_request_context

from src.config import config

# Add request_id filter
class RequestIdFilter(logging.Filter):
    """Add request_id to log records."""
    
    def filter(self, record):
        if has_request_context():
            record.request_id = getattr(g, 'request_id', 'no_request_id')
        else:
            record.request_id = 'no_request_id'
        return True

# Configure logger
def setup_logger(name="nyai"):
    """Configure logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create handlers
    file_handler = RotatingFileHandler(
        config.LOG_FILE, 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Add filter to handlers
    request_id_filter = RequestIdFilter()
    file_handler.addFilter(request_id_filter)
    console_handler.addFilter(request_id_filter)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Generate a request ID
def generate_request_id():
    """Generate a unique request ID."""
    return str(uuid.uuid4())

# Decorator to log function calls with timing
def log_function_call(logger):
    """Decorator to log function calls with timing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            # Generate function signature for logging
            func_args = ", ".join([str(arg) for arg in args])
            func_kwargs = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            func_signature = f"{func.__name__}({func_args}, {func_kwargs})"
            
            # Log function call
            logger.debug(f"Calling {func_signature}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                logger.debug(f"Finished {func.__name__} in {execution_time:.2f}ms")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Create and export logger
logger = setup_logger() 