from flask import Blueprint, request, jsonify, g
import uuid
import time

from src.config import config
from src.middleware import rate_limit, auth_required
from src.utils.logger import logger
from src.utils.errors import ValidationError, NotFoundError
from src.services.ai_service_refactored import legal_ai_service

# Create blueprint
ai_routes = Blueprint("ai_routes", __name__)

@ai_routes.route("/generate", methods=["POST"])
@rate_limit()
@auth_required
def generate():
    """
    Generate AI response for a legal query.
    
    Accepts a JSON payload with:
    - prompt: The user's legal query
    - session_id: (Optional) Session ID for conversation history
    
    Returns:
    - JSON with AI response and session ID
    """
    # Start request timing
    start_time = time.time()
    
    # Get request data
    data = request.json
    if not data:
        raise ValidationError("Missing request body")
        
    if "prompt" not in data:
        raise ValidationError("Missing 'prompt' field")
    
    # Get or create session ID for conversation context
    session_id = data.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session: {session_id}")
    
    # Get response
    response = legal_ai_service.generate_response(
        prompt=data["prompt"], 
        session_id=session_id,
        request_id=g.get('request_id')
    )
    
    # Log response time
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Generated response in {duration_ms}ms")
    
    return jsonify({
        "reply": response,
        "session_id": session_id,
        "processing_time_ms": duration_ms
    })

@ai_routes.route("/translate", methods=["POST"])
@rate_limit()
@auth_required
def translate():
    """
    Translate text between languages.
    
    Accepts a JSON payload with:
    - input_sentences: List of sentences to translate
    - source_language: (Optional) Source language
    - target_language: Target language
    
    Returns:
    - JSON with translated text
    """
    # Get request data
    data = request.json
    if not data:
        raise ValidationError("Missing request body")
        
    if "input_sentences" not in data:
        raise ValidationError("Missing 'input_sentences' field")
    
    # TODO: Replace with your translation service implementation
    from src.services.translation_service import translate_sentences
    translation_result = translate_sentences(data["input_sentences"])
    
    return jsonify(translation_result)

@ai_routes.route("/clear-cache", methods=["POST"])
@auth_required
def cache_clear():
    """
    Clear the response cache and optionally conversation history.
    
    Accepts a JSON payload with:
    - clear_history: (Optional) Whether to clear conversation history
    - session_id: (Optional) Specific session ID to clear
    
    Returns:
    - JSON with status message
    """
    data = request.json or {}
    
    # Clear cache
    legal_ai_service.clear_cache()
    
    # Optionally clear conversation history for specific session or all sessions
    if data.get("clear_history", False):
        session_id = data.get("session_id")
        legal_ai_service.clear_conversation_history(session_id)
        return jsonify({"status": "Cache and conversation history cleared"})
    
    return jsonify({"status": "Cache cleared"})

@ai_routes.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns:
    - JSON with service status
    """
    # Check if LLM API is configured
    api_configured = bool(config.GOOGLE_API_KEY)
    
    # Basic health details
    health_details = {
        "status": "ok" if api_configured else "degraded",
        "service": "nyai-legal-assistant",
        "version": config.VERSION,
        "llm_api_configured": api_configured,
        "timestamp": time.time()
    }
    
    # Return detailed health check
    return jsonify(health_details)

@ai_routes.route("/sessions/<session_id>", methods=["DELETE"])
@auth_required
def delete_session(session_id):
    """
    Delete a specific conversation session.
    
    Path parameters:
    - session_id: ID of the session to delete
    
    Returns:
    - JSON with status message
    """
    if not session_id:
        raise ValidationError("Missing session ID")
    
    # Check if session exists
    if session_id not in legal_ai_service.conversation_histories:
        raise NotFoundError(f"Session {session_id} not found")
    
    # Clear conversation history for session
    legal_ai_service.clear_conversation_history(session_id)
    
    return jsonify({"status": "success", "message": f"Session {session_id} deleted"})

# Add more routes as needed
