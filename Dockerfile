FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOME=/home/user

RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY --chown=user ./requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models (keeps the Space from timing out on first run)
RUN python3 -c 'from transformers import pipeline; pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")'

# Copy code
COPY --chown=user . .

# CRITICAL: Grant full permissions to the app directory so the .db can be initialized
USER root
RUN chmod -R 777 $HOME/app
USER user

EXPOSE 7860

# Start with the host 0.0.0.0 to make it accessible
CMD ["python", "main.py"]
