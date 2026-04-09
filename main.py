"""
CritiqueConnect Backend — FastAPI Application

Hybrid NLP Pipeline (per published paper):
  Stage 1: Preprocessing (normalize, tokenize, noise removal, context filter, embeddings)
  Stage 2: BERT Semantic Analysis (sentiment, intent, clustering, quality scoring)
  Stage 3: Gemini Generative Synthesis (cluster summarization, recommendations, balanced perspective)
  Stage 4: Structured Final Report (digest, strengths, weaknesses, ranked recommendations)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from dotenv import load_dotenv

from database import Database
from collector import FeedbackCollector
from preprocessor import FeedbackPreprocessor
from analyzer import BERTAnalyzer
from enhancer import GPTEnhancer
from synthesizer import Synthesizer
from Agent.google import Gemini

# ──────────────────────────────────────────────
#  App Initialization
# ──────────────────────────────────────────────

app = FastAPI(
    title="CritiqueConnect API",
    description="AI-driven semantic analysis and adaptive feedback synthesis",
    version="2.0.0",
)

# Configure CORS (configurable via env)
load_dotenv()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize shared components ──
# Single Database instance (singleton) shared by all components
db = Database()

# Gemini agent (nullable — graceful degradation if API key missing)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  WARNING: GOOGLE_API_KEY not set. Gemini-powered endpoints will use fallbacks.")
    gemini_agent = None
else:
    gemini_agent = Gemini(GOOGLE_API_KEY)
    print("✅ Gemini agent initialized")

# Pipeline components with shared DB and Gemini agent
preprocessor = FeedbackPreprocessor()
collector = FeedbackCollector(db=db)
analyzer = BERTAnalyzer(db=db)
enhancer = GPTEnhancer(gemini_agent=gemini_agent, db=db)
synthesizer = Synthesizer(gemini_agent=gemini_agent, db=db)


# ──────────────────────────────────────────────
#  Pydantic Models
# ──────────────────────────────────────────────

class Work(BaseModel):
    user_id: str
    content: str
    type: str


class Critique(BaseModel):
    aspect: str
    raw_text: str


class WorkWithCritiques(BaseModel):
    work: Work
    critiques: List[Critique]


class AgentRequest(BaseModel):
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    reviews: List[str]


# ──────────────────────────────────────────────
#  Utility: Sanitize text for LLM prompts
# ──────────────────────────────────────────────

def sanitize_for_prompt(text: str, max_length: int = 2000) -> str:
    """Strip control characters and truncate to prevent prompt injection."""
    if not text:
        return ""
    cleaned = text.replace("```", "").replace("---", "")
    return cleaned[:max_length]


# ──────────────────────────────────────────────
#  Full Pipeline: Process a single critique through Stages 1-3
# ──────────────────────────────────────────────

def process_critique_pipeline(critique_id: int, raw_text: str, aspect: str = "") -> dict:
    """
    Run the full Critique Connect pipeline on a single critique:
      Stage 1: Preprocess
      Stage 2: Analyze (sentiment, intent, quality)
      Stage 3: Enhance (Gemini generative synthesis)
    """
    # ── Stage 1: Preprocessing ──
    preprocessed = preprocessor.preprocess(raw_text)

    # Store preprocessing results
    db.update_critique_preprocessing(
        critique_id,
        cleaned_text=preprocessed["cleaned_text"],
        is_meaningful=preprocessed["is_meaningful"],
    )

    # Skip non-meaningful feedback (context filter)
    if not preprocessed["is_meaningful"]:
        return {
            "critique_id": critique_id,
            "is_meaningful": False,
            "message": "Feedback filtered as non-actionable (too generic or empty)",
            "scores": None,
            "enhanced_text": None,
        }

    # ── Stage 2: Semantic Analysis ──
    analysis = analyzer.analyze_and_store(critique_id)

    # ── Stage 3: Generative Synthesis (Enhancement) ──
    enhanced_text = enhancer.enhance_and_store(critique_id)

    return {
        "critique_id": critique_id,
        "is_meaningful": True,
        "preprocessing": {
            "cleaned_text": preprocessed["cleaned_text"],
            "token_count": len(preprocessed["tokens"]),
        },
        "analysis": analysis,
        "enhanced_text": enhanced_text,
        # Backward-compatible fields
        "scores": {
            "tone": analysis["tone_score"] if analysis else None,
            "actionability": analysis["actionability_score"] if analysis else None,
        } if analysis else None,
    }


# ──────────────────────────────────────────────
#  API Endpoints
# ──────────────────────────────────────────────

@app.post("/api/works")
async def create_work(work: Work):
    """Create a new creative work."""
    work_id = collector.add_work(work.user_id, work.content, work.type)
    if not work_id:
        raise HTTPException(status_code=500, detail="Failed to create work")
    return {"work_id": work_id}


@app.post("/api/works/{work_id}/critiques")
async def add_critique(work_id: int, critique: Critique):
    """
    Add a critique and process it through the full pipeline.
    Stage 1 → Stage 2 → Stage 3
    """
    # Add critique to database
    critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
    if not critique_id:
        raise HTTPException(status_code=500, detail="Failed to add critique")

    # Run full pipeline
    result = process_critique_pipeline(critique_id, critique.raw_text, critique.aspect)

    return result


@app.get("/api/works/{work_id}")
async def get_work(work_id: int):
    """Get a work and its critiques with full analysis details."""
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    # Use the detailed query that returns dicts
    critiques = db.get_all_critique_details_for_work(work_id)

    return {
        "work": {
            "id": work[0],
            "user_id": work[1],
            "content": work[2],
            "type": work[3],
            "created_at": work[4],
        },
        "critiques": critiques,
    }


@app.get("/api/works/{work_id}/synthesize")
async def synthesize_work_critiques(work_id: int):
    """
    Synthesize all critiques for a work into a summary.
    Backward-compatible endpoint (returns summary string).
    """
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    report = synthesizer.synthesize_work_critiques(work_id)

    return {
        "work_id": work_id,
        "summary": report.get("digest", "No critiques to synthesize."),
    }


@app.get("/api/works/{work_id}/report")
async def generate_report(work_id: int):
    """
    Generate the full Critique Connect structured report (Paper Stage 4).

    Returns:
        - digest: Overall summary
        - sentiment_overview: Positive/neutral/negative percentages
        - strengths: List of identified strengths
        - weaknesses: List of identified weaknesses
        - recommendations: Ranked actionable recommendations
        - critique_count: Total number of analysed critiques
        - average_quality: Mean quality score of all critiques
    """
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    # Get analysed critiques
    critiques_data = db.get_enhanced_critiques_for_work(work_id)

    if not critiques_data:
        raise HTTPException(status_code=404, detail="No analysed critiques found for this work")

    # Perform thematic clustering (Stage 2 — sub-process 3)
    embeddings = []
    critique_ids = []
    for c in critiques_data:
        text = c.get("cleaned_text") or c.get("raw_text", "")
        emb = preprocessor.vectorize(text)
        embeddings.append(emb)
        critique_ids.append(c["id"])

    if len(embeddings) >= 2:
        clustering = analyzer.cluster_themes(embeddings)
        # Store cluster assignments
        for cid, cluster_label in zip(critique_ids, clustering["labels"]):
            db.update_critique_cluster(cid, cluster_label)
        # Update critiques_data with cluster info
        for c, label in zip(critiques_data, clustering["labels"]):
            c["cluster_id"] = label
    else:
        clustering = {"labels": [0], "silhouette_score": 0.0, "n_clusters": 1}

    # Generate structured report (Stages 3-4)
    report = synthesizer.synthesize_critiques(critiques_data)

    # Add clustering info
    report["clustering"] = {
        "n_clusters": clustering["n_clusters"],
        "silhouette_score": clustering["silhouette_score"],
    }

    return {
        "work_id": work_id,
        "work_content": work[2],
        "work_type": work[3],
        "report": report,
    }


@app.post("/api/process-feedback")
async def process_feedback(data: WorkWithCritiques):
    """
    Process a work and its critiques through the full pipeline in one go.
    Runs Stage 1 → Stage 2 → Stage 3 for each critique, then Stage 4 synthesis.
    """
    # Create work
    work_id = collector.add_work(data.work.user_id, data.work.content, data.work.type)
    if not work_id:
        raise HTTPException(status_code=500, detail="Failed to create work")

    results = []
    for critique in data.critiques:
        # Add critique to database
        critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
        if not critique_id:
            continue

        # Run full pipeline
        result = process_critique_pipeline(critique_id, critique.raw_text, critique.aspect)
        results.append(result)

    # Stage 4: Generate structured report
    critiques_data = db.get_enhanced_critiques_for_work(work_id)
    report = synthesizer.synthesize_critiques(critiques_data) if critiques_data else None

    return {
        "work_id": work_id,
        "results": results,
        "report": report,
        # Backward compatible
        "summary": report.get("digest") if report else None,
    }


@app.post("/api/agent/process")
async def process_with_agent(request: AgentRequest):
    """
    Process input data through the Gemini agent for direct review analysis.
    Returns structured output matching the paper's report format.
    """
    if gemini_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini agent not configured. Set GOOGLE_API_KEY in .env"
        )

    # Sanitize inputs
    title = sanitize_for_prompt(request.title)
    description = sanitize_for_prompt(request.description or "Not provided")
    reviews = [sanitize_for_prompt(r) for r in request.reviews]

    prompt = f"""You are an expert review analyst using the Critique Connect framework.

Analyze the following user reviews for the given product or service, and provide a
structured response matching the Critique Connect report format.

Title: {title}
Description: {description}

Reviews:
{chr(10).join('- ' + review for review in reviews)}

Based on the reviews, return the response in the following JSON format:

{{
    "id": "{request.id}",
    "overall_summary": "<A concise and insightful summary of the overall user feedback>",
    "improvement_points": [
        "<List specific points where users suggest or imply improvements are needed>"
    ],
    "sentiment_analysis": {{
        "positive": "<% of reviews that are positive>",
        "neutral": "<% of reviews that are neutral>",
        "negative": "<% of reviews that are negative>"
    }}
}}

Guidelines:
- Extract patterns and recurring themes from the reviews.
- Use a critical but fair tone, like a professional reviewer.
- Make the summary insightful and actionable.
- Ensure sentiment percentages reflect the tone and content of the reviews.
- Balance strengths and weaknesses for a fair assessment.

Output valid JSON only, without any code blocks or other text."""

    try:
        response = gemini_agent.generate(prompt)
        raw_text = response.text
        cleaned_output = raw_text.strip().strip("` \n")

        # Remove "json" header if present
        if cleaned_output.startswith("json"):
            cleaned_output = "\n".join(cleaned_output.split("\n")[1:]).strip()

        # Parse the cleaned JSON
        data = json.loads(cleaned_output)

        return data
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse agent response as JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request through agent: {str(e)}"
        )


# ──────────────────────────────────────────────
#  Health / Utility Endpoints
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Welcome to CritiqueConnect API",
        "version": "2.0.0",
        "pipeline": "Hybrid: BERT (analysis) + Gemini (generation)",
    }


@app.get("/api/hello")
async def hello():
    return {"message": "Hello from CritiqueConnect!"}


# ──────────────────────────────────────────────
#  Lifecycle Events
# ──────────────────────────────────────────────

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown."""
    db.close()
    print("Database connection closed.")


# ──────────────────────────────────────────────
#  Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)