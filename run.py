from src.app import app
import os

if __name__ == "__main__":
    # Use environment variables for configuration if available
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    app.run(host=host, port=port, debug=debug)
