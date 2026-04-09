# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Create a non-root user (good practice for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies (needed for some ML libs)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# Install dependencies
# Using CPU-only torch to save space and memory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download ML models so the app starts faster
RUN python3 -c 'from transformers import pipeline; pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")' && \
    python3 -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2")'

# Copy the rest of the application code
COPY --chown=user . .

# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "main.py"]
