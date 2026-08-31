FROM python:3.12-slim

WORKDIR /app

# System deps some pip packages (faiss, sentence-transformers) need to build/run
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the app to listen on port 7860
EXPOSE 7860

CMD ["python", "app.py"]