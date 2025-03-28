from flask import Flask
from flask_cors import CORS
from src.routes.ai_routes import ai_routes

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(ai_routes, url_prefix="/api")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
