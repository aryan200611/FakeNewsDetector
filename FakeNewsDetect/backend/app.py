"""
STYLEPRINT backend API.

Serves the full-size (20k-feature) fake-news style classifier behind a
Flask API, so model weights never ship to the client.

Endpoints:
    GET  /health          -> {"status": "ok", "n_features": ..., "accuracy": ...}
    POST /predict          body: {"text": "..."}
                           -> {"verdict", "confidence", "probability_fake",
                                "matched_terms", "top_fake_terms", "top_real_terms"}

Run locally:
    python app.py
    # or for production: gunicorn -w 2 -b 0.0.0.0:8000 app:app
"""
import os
import re
import string

import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = 'model.joblib'
MAX_INPUT_CHARS = 20000  # basic guardrail against huge payloads

# Restrict this in production by setting the ALLOWED_ORIGIN env var to your
# deployed frontend's exact origin, e.g. https://yourusername.github.io
ALLOWED_ORIGIN = os.environ.get('ALLOWED_ORIGIN', '*')

app = Flask(__name__)

print(f"Loading model from {MODEL_PATH}...")
artifact = joblib.load(MODEL_PATH)
vectorizer = artifact['vectorizer']
model = artifact['model']
MODEL_ACCURACY = artifact.get('accuracy')
N_FEATURES = artifact.get('n_features')
FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())
COEF = model.coef_[0]
print(f"Model loaded: {N_FEATURES} features, held-out accuracy {MODEL_ACCURACY:.4f}")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def top_contributing_terms(tfidf_vector, top_n=6):
    """Given a single-row sparse TF-IDF vector, return top terms pushing
    toward FAKE and toward REAL, using the linear model's coefficients."""
    row = tfidf_vector.tocoo()
    contributions = []
    for idx, value in zip(row.col, row.data):
        contributions.append((FEATURE_NAMES[idx], float(value) * float(COEF[idx])))

    contributions.sort(key=lambda x: x[1], reverse=True)
    top_fake = [{'term': t, 'contribution': c} for t, c in contributions[:top_n] if c > 0]
    top_real = [{'term': t, 'contribution': c} for t, c in contributions[-top_n:] if c < 0]
    top_real.reverse()
    return top_fake, top_real, len(contributions)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGIN
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'n_features': N_FEATURES,
        'accuracy': MODEL_ACCURACY,
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json(silent=True) or {}
    text = data.get('text', '')

    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Field "text" is required and must be a non-empty string.'}), 400

    if len(text) > MAX_INPUT_CHARS:
        return jsonify({'error': f'Text too long. Max {MAX_INPUT_CHARS} characters.'}), 400

    cleaned = clean_text(text)
    tfidf_vector = vectorizer.transform([cleaned])

    probability_fake = float(model.predict_proba(tfidf_vector)[0][1])
    is_fake = probability_fake >= 0.5
    verdict = 'FAKE' if is_fake else 'REAL'
    confidence = probability_fake if is_fake else (1 - probability_fake)

    top_fake_terms, top_real_terms, matched_terms = top_contributing_terms(tfidf_vector)

    return jsonify({
        'verdict': verdict,
        'confidence': round(confidence, 4),
        'probability_fake': round(probability_fake, 4),
        'matched_terms': matched_terms,
        'top_fake_terms': top_fake_terms,
        'top_real_terms': top_real_terms,
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
