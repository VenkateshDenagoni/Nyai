# NyAI - Multilingual Legal AI Assistant

## Project Overview

NyAI is an AI-powered legal assistance platform focused on providing accurate legal insights for the Indian legal system in multiple Indian languages.

## Features

- **Legal Domain Expertise**: Specialized in Indian legal matters including consumer rights, property laws, business compliance, and more
- **Multilingual Support**: Fully supports English, Hindi, Tamil, Telugu, Marathi, and Bengali
- **Contextual Responses**: Remembers conversation history to provide relevant, personalized assistance
- **Production-Ready**: Includes error handling, rate limiting, and safety features

## Setup Instructions

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up environment variables:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   NYAI_ENV=development  # or production
   ```
6. Run the application: `python run.py`

## API Routes

- `POST /api/chat`: Main endpoint for conversational responses
- `POST /api/clear-cache`: Clear response cache and conversation history
- `GET /api/health`: Health check endpoint
- `DELETE /api/sessions/<session_id>`: Delete a specific conversation session

## Docker Support

Build and run the application using Docker:

```bash
docker build -t nyai-legal .
docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key nyai-legal
```

## Technologies

- Flask (Web Framework)
- Google Gemini Pro (LLM)
- Python 3.11+

## Language Support

NyAI supports the following languages:

- English (eng_Latn)
- Hindi (hin_Deva)
- Tamil (tam_Taml)
- Telugu (tel_Telu)
- Marathi (mar_Deva)
- Bengali (ben_Beng)

The assistant automatically detects the language of user queries and responds in the same language. It can also adapt to language changes mid-conversation.

## Contributors

D. Venkatesh
