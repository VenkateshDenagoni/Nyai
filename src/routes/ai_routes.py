from flask import Blueprint, request, jsonify
from src.services.ai_service import generate_response
from src.services.translation_service import translate_sentences

ai_routes = Blueprint("ai_routes", __name__)

@ai_routes.route("/generate", methods=["POST"])
def generate():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400
    return jsonify({"reply": generate_response(data["prompt"])})

@ai_routes.route("/translate", methods=["POST"])
def translate():
    data = request.json
    if not data or "input_sentences" not in data:
        return jsonify({"error": "Missing 'input_sentences'"}), 400
    return jsonify(translate_sentences(data["input_sentences"]))
