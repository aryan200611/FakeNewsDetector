# STYLEPRINT Backend

A Flask API that serves predictions from the full 20,000-feature fake-news
style-detector model. The model (vocabulary, IDF weights, coefficients)
lives only on the server — the frontend never sees any of it, only the
JSON result of a prediction.

Trained on the Kaggle Fake and Real News Dataset (~39k articles after
cleaning). Held-out accuracy: ~98% (see the notebook's Section 9 for a more
honest cross-subject generalization number, ~93%, and the important caveats
about what this model actually detects — writing style, not truthfulness).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API starts at `http://localhost:8000`. A pre-trained model is already
included as `model.joblib` — you don't need to retrain anything to run this.

## Endpoints

### `GET /health`
```json
{"status": "ok", "n_features": 20000, "accuracy": 0.9794}
```

### `POST /predict`
Request:
```json
{"text": "Your article text here..."}
```

Response:
```json
{
  "verdict": "FAKE",
  "confidence": 0.94,
  "probability_fake": 0.94,
  "matched_terms": 7,
  "top_fake_terms": [{"term": "breaking", "contribution": 1.2}, ...],
  "top_real_terms": [{"term": "wont", "contribution": -0.08}, ...]
}
```

Errors return `400` with `{"error": "..."}` for empty/missing text or text
over 20,000 characters.

## Frontend

The `../frontend/styleprint.html` file is already wired up to call this API
— it fetches `http://localhost:8000/predict` and renders the verdict,
confidence gauge, and contributing terms. Just start this backend, then open
that HTML file directly in a browser. No build step required.

If you deploy this backend somewhere other than `localhost:8000`, update the
`API_BASE_URL` constant near the top of the `<script>` block in
`styleprint.html` to match.

## Deploying to Render (free tier)

This repo includes `render.yaml` and a `Procfile`, so deployment is mostly
point-and-click:

1. Push this repo to GitHub (see the top-level README for git commands).
2. Go to [render.com](https://render.com) and sign up / log in (GitHub login
   is easiest).
3. Click **New +** → **Web Service** → connect your GitHub account → select
   this repo.
4. Render should auto-detect `render.yaml` and pre-fill the settings
   (Python, `pip install -r requirements.txt`, gunicorn start command). If
   it asks you to set the **Root Directory**, set it to `backend`.
5. Click **Create Web Service**. The first deploy takes a few minutes (it's
   loading a 954 KB model file — trivial by ML standards, but give it a
   moment).
6. Once live, Render gives you a URL like
   `https://styleprint-backend-xxxx.onrender.com`. Test it:
   ```bash
   curl https://styleprint-backend-xxxx.onrender.com/health
   ```
7. **Copy that URL** — you'll paste it into the frontend (see top-level
   README's frontend deployment section).

**Free tier note:** Render's free web services spin down after inactivity
and take ~30-60 seconds to wake up on the next request. That's normal, not
a bug — the first request after idle time will just be slow.

**Locking down CORS:** once you know your frontend's exact deployed URL
(e.g. `https://yourusername.github.io`), set an environment variable in
Render's dashboard: `ALLOWED_ORIGIN` = that exact URL. This restricts the
API to only respond to requests from your frontend instead of any website.

## Retraining the model

If you want to retrain (e.g. with more features, a different dataset, or
updated cleaning rules):

```bash
python train_model.py --fake Fake.csv --true True.csv --out model.joblib --max-features 20000
```

Requires the Kaggle Fake and Real News Dataset CSVs (`Fake.csv`, `True.csv`)
in the working directory, or pass paths via `--fake`/`--true`.

## Production notes

- The built-in `python app.py` server is for development only. For
  production, use gunicorn (already configured in `Procfile`/`Dockerfile`):
  ```bash
  gunicorn -w 2 -b 0.0.0.0:$PORT app:app
  ```
- CORS defaults to wide open (`Access-Control-Allow-Origin: *`) for ease of
  local development. Set the `ALLOWED_ORIGIN` environment variable to your
  frontend's exact deployed URL before treating this as production-hardened
  (see the Render deployment section above).
- No rate limiting or auth is included. Add these (e.g. Flask-Limiter, an
  API key check) before exposing this publicly to avoid abuse.
- Suggested free/cheap hosts for a small Flask API like this: Render,
  Railway, Fly.io, or a small VPS. A `Dockerfile` is included for any of
  these.
