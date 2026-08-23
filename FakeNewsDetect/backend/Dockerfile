FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.joblib ./

# Most hosting platforms (Render, Railway, Fly.io) inject $PORT at runtime.
# Default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE $PORT

CMD gunicorn -w 2 -b 0.0.0.0:$PORT app:app
