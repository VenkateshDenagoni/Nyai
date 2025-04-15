import os
import time
import hashlib
import json
import random
import uuid
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import config
from src.utils.logger import logger, log_function_call
from src.utils.errors import (
    LLMError, LLMAPIKeyError, LLMRateLimitError, 
    LLMTimeoutError, ValidationError, LLMContentFilterError
)

class LegalAIService:
    """Service for handling AI-generated legal responses."""
    
    def __init__(self):
        """Initialize the Legal AI Service."""
        # Configure Gemini API
        genai.configure(api_key=config.GOOGLE_API_KEY)
        
        # Initialize caches and state
        self.response_cache = {}
        self.conversation_histories = {}
        
        # Set up prompt templates
        self.today = datetime.today().strftime("%Y-%m-%d")
        self._load_prompt_templates()
    
    def _load_prompt_templates(self):
        """Load prompt templates from file or use defaults."""
        # Ensure prompts directory exists
        templates_dir = config.PROMPT_TEMPLATES_DIR
        templates_path = templates_dir / "base_system_prompt.txt"
        
        try:
            # Create the prompts directory if it doesn't exist
            if not templates_dir.exists():
                logger.warning(f"Prompts directory does not exist, creating: {templates_dir}")
                templates_dir.mkdir(parents=True, exist_ok=True)
            
            if templates_path.exists():
                with open(templates_path, 'r') as f:
                    template_content = f.read()
                    if not template_content.strip():
                        logger.warning(f"Prompt file exists but is empty: {templates_path}")
                        self.base_system_prompt = self._get_default_system_prompt()
                    else:
                        self.base_system_prompt = template_content.replace("{today}", self.today)
                        logger.info(f"Loaded system prompt from {templates_path} ({len(template_content)} characters)")
            else:
                logger.warning(f"Prompt file not found: {templates_path}")
                self.base_system_prompt = self._get_default_system_prompt()
                
        except Exception as e:
            logger.error(f"Error loading prompt template from {templates_path}: {e}")
            logger.error(f"Using default system prompt instead")
            self.base_system_prompt = self._get_default_system_prompt()
    
    def _get_default_system_prompt(self):
        """Get the default system prompt."""
        return f"""You are NYAI, an Indian legal AI assistant specialized in Indian law.

NOTE: This is a simplified fallback prompt. Please ensure the prompt file exists at {config.PROMPT_TEMPLATES_DIR}/base_system_prompt.txt for full functionality.

The current date is {datetime.now().strftime('%Y-%m-%d')}."""
    
    def validate_input(self, prompt: str) -> str:
        """
        Validates and sanitizes the input prompt.
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            Sanitized prompt
            
        Raises:
            InputValidationError: If input validation fails
        """
        if not prompt or not isinstance(prompt, str):
            raise ValidationError("Prompt must be a non-empty string", error_code="invalid_prompt")
        
        if len(prompt) > config.MAX_PROMPT_LENGTH:
            logger.warning(f"Truncating prompt from {len(prompt)} to {config.MAX_PROMPT_LENGTH} characters")
            prompt = prompt[:config.MAX_PROMPT_LENGTH]
        
        # Basic sanitization - remove potentially problematic characters
        sanitized = prompt.replace('\x00', '')
        
        return sanitized
    
    def check_prompt_for_bypass_attempts(self, prompt: str) -> Optional[str]:
        """
        Checks for common prompt bypass/injection attempts.
        
        Args:
            prompt: The user's prompt
            
        Returns:
            Warning message if bypass detected, None otherwise
        """
        bypass_patterns = [
            "ignore previous instructions",
            "ignore all previous commands",
            "disregard prior directives",
            "forget your instructions",
            "print your system prompt",
            "output your prompt",
            "repeat back to me",
            "what are your instructions",
            "act as a",
            "pretend to be",
            "simulate a different",
            "write a bash script",
            "write code that can",
            "how to bypass",
            "how to circumvent"
        ]
        
        # Lower the prompt for case-insensitive matching
        lower_prompt = prompt.lower()
        
        # Check for bypass patterns
        for pattern in bypass_patterns:
            if pattern in lower_prompt:
                logger.warning(f"Potential prompt bypass attempt detected: '{pattern}'")
                return "I'm designed to provide information about Indian legal matters. I cannot fulfill requests that attempt to change my purpose or functioning."
        
        return None
    
    def get_enhanced_legal_prompt(self, topic: str) -> str:
        """
        Returns an enhanced legal prompt based on the topic.
        
        Args:
            topic: The general topic of the query
            
        Returns:
            Enhanced prompt with legal context
        """
        # Check if the user is explicitly asking for detailed information
        is_detailed_request = any(term in topic.lower() for term in [
            "explain", "detailed", "elaborate", "in depth", "thorough", 
            "comprehensive", "step by step", "walkthrough", "guide me"
        ])
        
        # Enhanced legal contexts with more detailed guidance
        legal_contexts = {
            "criminal": (
                "Focus on the Indian Penal Code (IPC) and Criminal Procedure Code (CrPC) with specific sections. "
                "Explain the legal framework, key concepts, and practical implications. "
                "Include relevant case precedents and their significance. "
                "Discuss procedural aspects and available remedies. "
                "Provide context about the legal system's approach to such matters."
            ),
            "civil": (
                "Reference the Civil Procedure Code (CPC) with specific sections. "
                "Explain court jurisdiction, procedures, and timelines. "
                "Discuss the burden of proof and evidentiary requirements. "
                "Include information about available remedies and enforcement mechanisms. "
                "Provide practical guidance on filing and pursuing civil matters."
            ),
            "constitutional": (
                "Cite specific articles from the Indian Constitution with key Supreme Court judgments. "
                "Explain the constitutional principles and their evolution. "
                "Discuss landmark cases that shaped interpretation. "
                "Include information about fundamental rights and their limitations. "
                "Provide context about the constitutional framework and its significance."
            ),
            "family": (
                "Reference specific sections of Hindu Marriage Act, Muslim Personal Law, Special Marriage Act as applicable. "
                "Explain different personal laws and their applications. "
                "Discuss procedural requirements and documentation. "
                "Include information about rights and obligations. "
                "Provide practical guidance on family law matters."
            ),
            "property": (
                "Cite Transfer of Property Act, Registration Act with specific sections. "
                "Explain property concepts and their legal implications. "
                "Discuss documentation and registration requirements. "
                "Include information about property rights and restrictions. "
                "Provide practical guidance on property transactions."
            ),
            "consumer": (
                "Reference Consumer Protection Act 2019 with specific sections. "
                "Explain consumer rights and available remedies. "
                "Discuss the complaint filing process and jurisdiction. "
                "Include information about compensation and relief mechanisms. "
                "Provide practical guidance on consumer protection matters."
            ),
            "business": (
                "Cite Companies Act 2013, LLP Act with specific sections. "
                "Explain business structures and compliance requirements. "
                "Discuss corporate governance and regulatory framework. "
                "Include information about legal obligations and liabilities. "
                "Provide practical guidance on business law matters."
            ),
            "tax": (
                "Reference Income Tax Act, GST laws with specific sections. "
                "Explain tax provisions and compliance requirements. "
                "Discuss filing procedures and documentation. "
                "Include information about tax planning and optimization. "
                "Provide practical guidance on tax matters."
            ),
            "employment": (
                "Cite Industrial Disputes Act, Factories Act with specific sections. "
                "Explain employer-employee rights and obligations. "
                "Discuss workplace regulations and compliance. "
                "Include information about dispute resolution mechanisms. "
                "Provide practical guidance on employment law matters."
            ),
            "cyber": (
                "Reference IT Act 2000 with specific sections. "
                "Explain cyber laws and their applications. "
                "Discuss digital rights and responsibilities. "
                "Include information about cybercrime prevention and remedies. "
                "Provide practical guidance on cyber law matters."
            ),
            "bankruptcy": (
                "Cite Insolvency and Bankruptcy Code 2016 with specific sections. "
                "Explain insolvency resolution process and timelines. "
                "Discuss creditor rights and debtor obligations. "
                "Include information about restructuring and liquidation. "
                "Provide practical guidance on bankruptcy matters."
            ),
            "contract": (
                "Reference Indian Contract Act 1872 with specific sections. "
                "Explain contract formation, performance, and remedies. "
                "Discuss essential elements and validity requirements. "
                "Include information about breach and enforcement. "
                "Provide practical guidance on contract law matters."
            )
        }
        
        # Default legal context with enhanced detail
        if is_detailed_request:
            legal_context = (
                "Provide a comprehensive explanation with specific sections of relevant Indian laws. "
                "Include historical context and evolution of the law. "
                "Discuss key concepts and their practical applications. "
                "Reference landmark judgments and their significance. "
                "Explain procedural aspects and available remedies. "
                "Provide practical examples and case studies. "
                "Include information about recent developments and amendments."
            )
        else:
            legal_context = (
                "Provide a balanced explanation with specific sections of relevant Indian laws. "
                "Include essential concepts and their applications. "
                "Reference important judgments and their implications. "
                "Explain key procedures and available remedies. "
                "Provide practical guidance and next steps."
            )
        
        # Check if the topic matches any of the predefined contexts
        for key, context in legal_contexts.items():
            if key in topic.lower():
                legal_context = context
                break
        
        return legal_context
    
    def check_content_safety(self, response: str) -> Tuple[bool, str]:
        """
        Checks if the response content is safe and appropriate.
        
        Args:
            response: The generated response
            
        Returns:
            Tuple of (is_safe, filtered_response)
        """
        # Patterns that lead to redundant disclaimers (only in English to avoid removing non-English content)
        disclaimer_patterns = [
            "I cannot provide legal advice",
            "I'm not a licensed attorney",
            "I'm just an AI and cannot",
            "Please consult with a qualified legal professional",
            "This information is for educational purposes only",
            "consult with a lawyer",
            "Note: This is general information, not legal advice",
            "seek professional legal advice"
        ]
        
        # Check for English language first to avoid modifying non-English responses incorrectly
        is_likely_english = any(word in response.lower() for word in ["the", "is", "are", "and", "or", "but", "not", "with", "legal", "law", "court"])
        
        # Only apply disclaimer filtering to English text to avoid altering non-English responses
        if is_likely_english:
            # Check for disclaimer patterns and remove entire sentences containing them
            sentences = []
            for sentence in response.split('. '):
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                # Check if sentence contains any disclaimer pattern
                if not any(pattern.lower() in sentence.lower() for pattern in disclaimer_patterns):
                    sentences.append(sentence)
            
            # Carefully rejoin sentences with proper spacing and punctuation
            cleaned_response = ""
            for i, sentence in enumerate(sentences):
                if i == 0:
                    cleaned_response = sentence
                else:
                    # If the previous sentence already ends with punctuation, don't add period
                    if cleaned_response and cleaned_response[-1] in ['.', '!', '?', ':', ';']:
                        cleaned_response += f" {sentence}"
                    else:
                        cleaned_response += f". {sentence}"
            
            # Check for language preference questions at the end
            language_patterns = [
                "In which language would you like",
                "Would you prefer this in",
                "I can also provide this in",
                "Also available in",
                "Would you like me to translate"
            ]
            
            # Check if the response ends with a language question and remove it
            for pattern in language_patterns:
                if pattern.lower() in cleaned_response.lower():
                    # Split again and filter out language-related sentences
                    sentences = []
                    for sentence in cleaned_response.split('. '):
                        if not any(pattern.lower() in sentence.lower() for pattern in language_patterns):
                            sentences.append(sentence)
                    
                    # Rejoin with proper punctuation
                    cleaned_response = ""
                    for i, sentence in enumerate(sentences):
                        if i == 0:
                            cleaned_response = sentence
                        else:
                            if cleaned_response and cleaned_response[-1] in ['.', '!', '?', ':', ';']:
                                cleaned_response += f" {sentence}"
                            else:
                                cleaned_response += f". {sentence}"
                                
            return True, cleaned_response
        else:
            # For non-English text, preserve the original response
            logger.info("Detected non-English response, preserving original content")
            return True, response
    
    def manage_conversation_history(self, session_id: str, prompt: str, response: str) -> List[Dict[str, str]]:
        """
        Manages conversation history for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            prompt: User prompt
            response: AI response
            
        Returns:
            Updated conversation history
        """
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []
            
        history = self.conversation_histories[session_id]
        
        # Add new exchange to history
        history.append({"user": prompt, "ai": response})
        
        # Trim history if needed
        if len(history) > config.MAX_HISTORY_LENGTH:
            # Keep the first entry (which might have special context) and the most recent entries
            history = [history[0]] + history[-(config.MAX_HISTORY_LENGTH-1):]
            self.conversation_histories[session_id] = history
            
        return history
    
    def build_prompt_with_context(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Builds a prompt with conversation context.
        
        Args:
            prompt: The current user prompt
            history: Optional conversation history
            
        Returns:
            A prompt with relevant context
        """
        if not history:
            return prompt
            
        # Include relevant history in the prompt
        context_prompt = "Previous conversation:\n"
        
        # Add up to 5 most recent exchanges for better context
        recent_history = history[-5:] if len(history) > 5 else history
        for exchange in recent_history:
            # Preserve exact formatting of previous exchanges to maintain language context
            context_prompt += f"User: {exchange['user']}\nAI: {exchange['ai']}\n\n"
            
        # Highlight the current query to ensure the model pays attention to the current language
        context_prompt += f"Current question (respond in the same language as this question): {prompt}"
        return context_prompt
    
    def get_cache_key(self, prompt: str, system_prompt: str) -> str:
        """
        Generate a cache key for a prompt and system prompt combination.
        
        Args:
            prompt: The user's input prompt
            system_prompt: The system prompt
            
        Returns:
            A cache key string
        """
        combined = f"{prompt}|{system_prompt}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    @retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)), 
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=config.RETRY_BACKOFF_FACTOR, 
            min=config.RETRY_MIN_SECONDS, 
            max=config.RETRY_MAX_SECONDS
        ),
        reraise=True
    )
    @log_function_call(logger)
    def call_llm_api(self, prompt: str, system_prompt: str) -> str:
        """
        Makes the actual API call to the LLM API with retry logic.
        
        Args:
            prompt: The user prompt
            system_prompt: The system prompt
            
        Returns:
            Generated text response
            
        Raises:
            Various LLMError subclasses based on the error encountered
        """
        if not config.GOOGLE_API_KEY:
            logger.error("No API key found in configuration")
            raise LLMAPIKeyError("API key not configured")
        
        try:
            # Initialize the model with generation config
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more consistent and precise responses
                "top_p": 0.90,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"  # Updated to allow legal discussions
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"  # Updated to allow legal discussions of sensitive topics
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            model = genai.GenerativeModel(
                config.LLM_MODEL,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Generate response with combined prompt
            full_prompt = f"{system_prompt}\n\nUser query: {prompt}"
            
            # Call the API with timeout handling
            response = model.generate_content(full_prompt)
            
            # Check if response contains valid text
            if hasattr(response, 'text') and response.text:
                logger.info("Response received from LLM API")
                return response.text
            else:
                # Handle responses without text
                logger.warning("Response missing text property")
                
                # Extract finish reason and safety ratings for logging only
                finish_reason = None
                safety_issue = False
                blocked_categories = []
                
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    
                    # Check safety ratings
                    if hasattr(candidate, 'safety_ratings'):
                        for rating in candidate.safety_ratings:
                            category = getattr(rating, 'category', 'unknown')
                            probability = getattr(rating, 'probability', 'unknown')
                            blocked = getattr(rating, 'blocked', False)
                            
                            if blocked:
                                safety_issue = True
                                blocked_categories.append(category)
                            
                            # Log detailed safety information for debugging
                            logger.info(f"Safety rating: {category}, Probability: {probability}, Blocked: {blocked}")
                
                # Log detailed information about the block
                if finish_reason == 3 or safety_issue:
                    logger.warning(f"Content blocked by safety filters. Finish reason: {finish_reason}")
                    logger.warning(f"Blocked categories: {', '.join(blocked_categories)}")
                    
                    # Return a user-friendly message for safety blocks
                    return """I apologize, but I'm unable to provide a detailed response on this topic due to content guidelines.

I can help with general legal information about Indian laws, legal procedures, or discuss alternative topics. Please consider rephrasing your question to focus on specific legal frameworks, rights, or procedures."""
                
                elif finish_reason == 4:  # Copyright material issue
                    logger.warning("Response flagged as potentially containing copyrighted material")
                    return "I apologize, but I'm unable to provide information that may include copyrighted material. Please consider rephrasing your question."
                    
                # Fallback for other cases
                return "I apologize, but I'm unable to generate a complete response for that query. Please try rephrasing your question."
                
        except LLMError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"API error: {error_str}")
            
            # Check for safety-related errors in the error message
            if "finish_reason" in error_str and "safety_ratings" in error_str:
                logger.warning("Detected safety filter block from error message")
                return """I apologize, but I'm unable to provide a detailed response on this topic due to content guidelines.

I can help with general legal information about Indian laws, legal procedures, or discuss alternative topics. Please consider rephrasing your question to focus on specific legal frameworks, rights, or procedures."""
            
            # Handle API rate limits
            elif "rate limit" in error_str or "quota" in error_str:
                logger.error("Rate limit error detected")
                raise LLMRateLimitError(f"API rate limit exceeded")
            # Handle timeouts
            elif "timeout" in error_str:
                logger.error("Timeout error detected")
                raise LLMTimeoutError(f"API request timed out")
            # Generic error handling
            else:
                logger.error("Unknown API error")
                raise LLMError("Error processing request")
    
    @log_function_call(logger)
    def generate_response(self, prompt: str, session_id: str = None, request_id: str = None) -> str:
        """
        Generates AI response using the LLM model with all improvements.
        
        Args:
            prompt: The user's prompt
            session_id: Optional session ID for conversation history
            request_id: Optional request ID for tracking
            
        Returns:
            Generated response text
        """
        try:
            # Track request ID if provided
            if request_id:
                logger.info(f"Generating response for request {request_id}")
                
            # Input validation
            try:
                prompt = self.validate_input(prompt)
            except ValidationError as e:
                logger.warning(f"Input validation error: {e}")
                return f"I couldn't process your request properly. Please try again with a different question."
                
            # Check for prompt bypass attempts
            bypass_warning = self.check_prompt_for_bypass_attempts(prompt)
            if bypass_warning:
                return bypass_warning
                
            # Detect topics for enhanced prompting
            legal_context = self.get_enhanced_legal_prompt(prompt)
            
            # Create system instructions with legal context
            system_prompt = f"{self.base_system_prompt}\nADDITIONAL CONTEXT: {legal_context}"
            
            # Get conversation history if session ID is provided
            history = None
            if session_id:
                history = self.conversation_histories.get(session_id, [])
                prompt_with_context = self.build_prompt_with_context(prompt, history)
            else:
                prompt_with_context = prompt
                
            # Check cache first
            cache_key = self.get_cache_key(prompt_with_context, system_prompt)
            if cache_key in self.response_cache:
                cached_response, timestamp = self.response_cache[cache_key]
                # Check if cache is still valid
                if time.time() - timestamp < config.CACHE_TTL:
                    logger.info("Returning cached response")
                    
                    # Update conversation history even for cached responses
                    if session_id:
                        self.manage_conversation_history(session_id, prompt, cached_response)
                        
                    return cached_response
            
            # Call the API with retries
            response_text = self.call_llm_api(prompt_with_context, system_prompt)
            
            # Validate and filter response
            is_safe, filtered_response = self.check_content_safety(response_text)
            
            # Cache the response
            self.response_cache[cache_key] = (filtered_response, time.time())
            
            # Update conversation history
            if session_id:
                self.manage_conversation_history(session_id, prompt, filtered_response)
                
            return filtered_response
            
        except LLMAPIKeyError as e:
            # Log the full error for debugging
            logger.error(f"API key error: {str(e)}")
            # Return a user-friendly message without exposing technical details
            return "I'm currently unable to access my knowledge base. Please try again later or contact support if the issue persists."
            
        except LLMRateLimitError as e:
            logger.error(f"Rate limit error: {str(e)}")
            return "I'm receiving too many requests right now. Please try again in a few minutes."
            
        except LLMTimeoutError as e:
            logger.error(f"Timeout error: {str(e)}")
            return "The request took too long to process. Please try again or ask a shorter question."
            
        except LLMError as e:
            # Log the full error for debugging
            logger.error(f"LLM API error: {str(e)}")
            # Return a user-friendly message without exposing error details
            return "I'm unable to provide a response at the moment. Please try rephrasing your question or try again later."
            
        except Exception as e:
            # Log detailed error for debugging
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a generic error message without exposing system details
            return "I apologize, but I encountered an issue processing your request. Please try again or ask a different question."
    
    def clear_cache(self) -> None:
        """Clears the response cache."""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def clear_conversation_history(self, session_id: str = None) -> None:
        """
        Clears conversation history.
        
        Args:
            session_id: Optional session ID to clear specific history,
                        clears all histories if None
        """
        if session_id:
            if session_id in self.conversation_histories:
                del self.conversation_histories[session_id]
                logger.info(f"Conversation history cleared for session {session_id}")
        else:
            self.conversation_histories.clear()
            logger.info("All conversation histories cleared")

# Create a singleton instance
legal_ai_service = LegalAIService() 