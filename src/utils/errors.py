from flask import jsonify
from werkzeug.exceptions import HTTPException

# Base API Error class
class APIError(Exception):
    """Base class for API errors with standardized response format."""
    status_code = 500
    error_code = "internal_error"
    message = "An unexpected error occurred."
    
    def __init__(self, message=None, error_code=None, status_code=None, details=None):
        if message:
            self.message = message
        if error_code:
            self.error_code = error_code
        if status_code:
            self.status_code = status_code
        self.details = details
        super().__init__(self.message)
    
    def to_response(self):
        """Convert error to standardized JSON response."""
        response = {
            "error": {
                "code": self.error_code,
                "message": self.message
            }
        }
        
        # Add error details if available
        if self.details:
            response["error"]["details"] = self.details
            
        return jsonify(response), self.status_code

# Specific API error types
class BadRequestError(APIError):
    """400 Bad Request Error"""
    status_code = 400
    error_code = "bad_request"
    message = "The request was invalid or cannot be served."

class UnauthorizedError(APIError):
    """401 Unauthorized Error"""
    status_code = 401
    error_code = "unauthorized"
    message = "Authentication is required and has failed or not been provided."

class ForbiddenError(APIError):
    """403 Forbidden Error"""
    status_code = 403
    error_code = "forbidden"
    message = "You don't have permission to access this resource."

class NotFoundError(APIError):
    """404 Not Found Error"""
    status_code = 404
    error_code = "not_found"
    message = "The requested resource was not found."

class RateLimitError(APIError):
    """429 Too Many Requests Error"""
    status_code = 429
    error_code = "rate_limit_exceeded"
    message = "Rate limit exceeded. Please try again later."

class ValidationError(BadRequestError):
    """Validation Error (400)"""
    error_code = "validation_error"
    message = "The request data failed validation."

# LLM-specific errors
class LLMError(APIError):
    """Base class for LLM-related errors."""
    status_code = 500
    error_code = "llm_error"
    message = "An error occurred with the language model service."

class LLMAPIKeyError(LLMError):
    """LLM API Key Error"""
    error_code = "llm_api_key_error"
    message = "Invalid or missing LLM API key."

class LLMRateLimitError(LLMError):
    """LLM Rate Limit Error"""
    status_code = 429
    error_code = "llm_rate_limit_error"
    message = "LLM service rate limit exceeded."

class LLMTimeoutError(LLMError):
    """LLM Timeout Error"""
    error_code = "llm_timeout_error"
    message = "LLM service request timed out."

class LLMContentFilterError(LLMError):
    """LLM Content Filter Error"""
    status_code = 400
    error_code = "llm_content_filter_error"
    message = "The request or response was filtered by the LLM service's content filter."

# Register error handler with Flask app
def register_error_handlers(app):
    """Register error handlers with Flask app."""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        return error.to_response()
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        # Convert default Flask/Werkzeug errors to our format
        api_error = APIError(
            message=error.description,
            error_code=f"http_{error.code}",
            status_code=error.code
        )
        return api_error.to_response()
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        # Convert any unhandled exception to our format
        # In production, you'd want to add monitoring/alerting here
        import traceback
        from src.utils.logger import logger
        
        logger.error(f"Unhandled exception: {str(error)}")
        logger.error(traceback.format_exc())
        
        return APIError().to_response() 