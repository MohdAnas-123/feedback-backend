---
title: CritiqueConnect
emoji: 📝
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CritiqueConnect Backend


This is the backend implementation of CritiqueConnect, a platform for enhancing creative feedback using AI models.

## System Architecture

1. **Feedback Collection**: Users upload work (e.g., a design description as text) and request critique. Peers submit raw feedback as text.
2. **BERT Analysis**: Pre-trained BERT processes each critique, scoring tone (constructive vs. harsh, 0-1 scale) and actionability (specific vs. generic, 0-1 scale).
3. **GPT Enhancement**: GPT refines raw feedback into actionable, positive output. Example: "Text is hard to read" becomes "Increase text contrast for better readability."
4. **Synthesis**: GPT aggregates multiple critiques into a concise summary, e.g., "Three users suggest improving text readability."

## Components

- **FeedbackCollector**: Stores user work and raw critiques (text inputs).
- **BERTAnalyzer**: Uses a pre-trained BERT model to score tone and actionability.
- **GPTEnhancer**: Refines raw feedback using GPT.
- **Synthesizer**: Merges multiple enhanced critiques into a summary.

## Installation

1. Clone this repository or download the code.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## API Endpoints

- `POST /api/works` - Create a new creative work
- `POST /api/works/{work_id}/critiques` - Add a critique to a work
- `GET /api/works/{work_id}` - Get a work and its critiques
- `GET /api/works/{work_id}/synthesize` - Synthesize all critiques for a work
- `POST /api/process-feedback` - Process a work and multiple critiques at once

## Running the Server

```bash
cd backend
python main.py
```

The server will run on http://localhost:8000.

## Example Usage

### Creating a Work

```python
import requests

response = requests.post(
    "http://localhost:8000/api/works",
    json={
        "user_id": "user123",
        "content": "A minimalist landing page with blue and white color scheme.",
        "type": "web_design"
    }
)

work_id = response.json()["work_id"]
```

### Adding a Critique

```python
response = requests.post(
    f"http://localhost:8000/api/works/{work_id}/critiques",
    json={
        "aspect": "color scheme",
        "raw_text": "The blue and white colors clash and look unprofessional."
    }
)

result = response.json()
print(f"Tone score: {result['scores']['tone']}")
print(f"Actionability score: {result['scores']['actionability']}")
print(f"Enhanced text: {result['enhanced_text']}")
```

### Synthesizing Critiques

```python
response = requests.get(f"http://localhost:8000/api/works/{work_id}/synthesize")
summary = response.json()["summary"]
print(f"Summary: {summary}")
``` 