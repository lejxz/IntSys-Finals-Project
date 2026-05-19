"""Flask web server for email fraud and prompt injection detection."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from model.classifier import predict_email

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Serve the single-page UI."""
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan_email():
    """Analyze submitted email content and return prediction scores."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please provide email text to scan."}), 400

    try:
        result = predict_email(text)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)