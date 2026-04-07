"""
Stage 2: Understanding Feedback — Semantic Analysis using BERT

Paper reference: "Once pre-processed, the text undergoes semantic analysis using a
BERT-based model... This stage involves four key sub-processes:
1. Sentiment Classification  2. Intent Recognition
3. Thematic Clustering       4. Quality Scoring"

Hybrid pipeline: BERT for analysis (this module) + Gemini for generation (enhancer).
"""

import numpy as np
from database import Database


class BERTAnalyzer:
    """
    Stage 2 of the Critique Connect pipeline.
    Multi-dimensional BERT-based semantic analysis of peer feedback.
    Models are lazy-loaded on first use for fast server startup.
    """

    # Intent categories as defined in the paper
    INTENT_LABELS = ["praise", "suggestion", "criticism", "question", "observation"]

    def __init__(self, db=None):
        """Initialize the analyzer. Models loaded lazily on first call."""
        self.db = db or Database()

        # Lazy-loaded models
        self._sentiment_pipeline = None
        self._intent_pipeline = None

    # ──────────────────────────────────────────────
    #  Lazy model loading
    # ──────────────────────────────────────────────

    @property
    def sentiment_pipeline(self):
        """Lazy-load sentiment analysis pipeline."""
        if self._sentiment_pipeline is None:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            print("Loaded sentiment model: nlptown/bert-base-multilingual-uncased-sentiment")
        return self._sentiment_pipeline

    @property
    def intent_pipeline(self):
        """Lazy-load intent recognition pipeline (zero-shot classification)."""
        if self._intent_pipeline is None:
            from transformers import pipeline
            self._intent_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            print("Loaded intent model: facebook/bart-large-mnli")
        return self._intent_pipeline

    # ──────────────────────────────────────────────
    #  Sub-process 1: Sentiment Classification
    # ──────────────────────────────────────────────

    def classify_sentiment(self, text: str) -> dict:
        """
        Evaluate feedback for polarity and intensity.

        Paper: "Each feedback comment is evaluated for polarity
        (positive, neutral, negative) and intensity."

        Uses nlptown/bert-base-multilingual-uncased-sentiment (1-5 stars).
        Maps to polarity + intensity.
        """
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # truncate for model
            label = result["label"]  # e.g., "4 stars"
            score = result["score"]  # confidence

            stars = int(label.split()[0])

            # Map 1-5 stars to polarity
            if stars >= 4:
                polarity = "positive"
            elif stars == 3:
                polarity = "neutral"
            else:
                polarity = "negative"

            # Map 1-5 stars to intensity (0-1 scale)
            intensity = (stars - 1) / 4.0  # 1→0.0, 5→1.0

            return {
                "polarity": polarity,
                "intensity": round(intensity, 3),
                "confidence": round(score, 3),
                "raw_stars": stars,
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {"polarity": "neutral", "intensity": 0.5, "confidence": 0.0, "raw_stars": 3}

    # ──────────────────────────────────────────────
    #  Sub-process 2: Intent Recognition
    # ──────────────────────────────────────────────

    def recognize_intent(self, text: str) -> dict:
        """
        Distinguish between types of feedback intents.

        Paper: "Using transformer-based contextual embeddings, CC distinguishes
        between types of feedback intents, enabling categorisation into
        pedagogically proper forms."

        Categories: praise, suggestion, criticism, question, observation
        """
        try:
            result = self.intent_pipeline(
                text[:512],
                candidate_labels=self.INTENT_LABELS,
                multi_label=False
            )

            # Build intent scores dict
            intent_scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                intent_scores[label] = round(score, 3)

            return {
                "primary_intent": result["labels"][0],
                "confidence": round(result["scores"][0], 3),
                "all_intents": intent_scores,
            }
        except Exception as e:
            print(f"Intent recognition error: {e}")
            return {
                "primary_intent": "observation",
                "confidence": 0.0,
                "all_intents": {label: 0.0 for label in self.INTENT_LABELS},
            }

    # ──────────────────────────────────────────────
    #  Sub-process 3: Thematic Clustering
    # ──────────────────────────────────────────────

    def cluster_themes(self, embeddings: list, n_clusters: int = None) -> dict:
        """
        Group semantically similar comments using clustering.

        Paper: "The system applies deep embedded clustering to group semantically
        similar comments, reducing redundancy."

        Uses KMeans with silhouette score optimization.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            embeddings_array = np.array(embeddings)

            if len(embeddings_array) < 2:
                return {"labels": [0] * len(embeddings), "silhouette_score": 0.0, "n_clusters": 1}

            # Auto-determine optimal number of clusters if not specified
            if n_clusters is None:
                max_k = min(len(embeddings_array), 5)
                best_score = -1
                best_k = 2

                for k in range(2, max_k + 1):
                    try:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = km.fit_predict(embeddings_array)
                        score = silhouette_score(embeddings_array, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except Exception:
                        continue

                n_clusters = best_k
            else:
                n_clusters = min(n_clusters, len(embeddings_array))

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            sil_score = silhouette_score(embeddings_array, labels) if n_clusters > 1 else 0.0

            return {
                "labels": labels.tolist(),
                "silhouette_score": round(sil_score, 3),
                "n_clusters": n_clusters,
            }
        except ImportError:
            print("Warning: scikit-learn not installed. Skipping clustering.")
            return {"labels": list(range(len(embeddings))), "silhouette_score": 0.0, "n_clusters": len(embeddings)}
        except Exception as e:
            print(f"Clustering error: {e}")
            return {"labels": [0] * len(embeddings), "silhouette_score": 0.0, "n_clusters": 1}

    # ──────────────────────────────────────────────
    #  Sub-process 4: Quality Scoring
    # ──────────────────────────────────────────────

    def score_quality(self, text: str, sentiment: dict = None) -> dict:
        """
        Score the constructiveness of feedback.

        Paper: "A reinforcement-based scoring model evaluates the constructiveness
        of each comment based on clarity, specificity, and tone."

        Uses heuristic analysis + sentiment info for scoring.
        """
        tokens = text.lower().split()
        num_words = len(tokens)

        # ── Clarity: well-formed, clear language ──
        # Longer, properly punctuated feedback scores higher
        has_punctuation = any(c in text for c in ".!?;:")
        avg_word_length = np.mean([len(w) for w in tokens]) if tokens else 0
        clarity = min(1.0, (num_words / 30) * 0.5 + (avg_word_length / 8) * 0.3 + (0.2 if has_punctuation else 0.0))

        # ── Specificity: concrete, detailed feedback ──
        # Look for specific indicators: numbers, comparisons, technical terms, examples
        specific_indicators = [
            r"\d+",                          # numbers
            r"(?:for example|e\.g\.|such as|like|instead|rather|consider)",  # examples
            r"(?:because|since|due to|reason)",  # reasoning
            r"(?:should|could|would|try|suggest|recommend)",  # actionable
            r"(?:color|font|layout|spacing|contrast|size|margin|align)",  # design terms
        ]
        specificity_hits = sum(1 for pattern in specific_indicators if __import__("re").search(pattern, text.lower()))
        specificity = min(1.0, specificity_hits / 3.0)

        # ── Tone: constructive vs. harsh ──
        if sentiment:
            # Map sentiment intensity to tone score
            # High intensity negative = harsh (low tone)
            # High intensity positive = encouraging (high tone)
            if sentiment["polarity"] == "negative":
                tone = max(0.1, 1.0 - sentiment["intensity"])
            elif sentiment["polarity"] == "positive":
                tone = min(1.0, 0.5 + sentiment["intensity"] * 0.5)
            else:
                tone = 0.6  # neutral is okay
        else:
            tone = 0.5

        # ── Overall quality: weighted combination ──
        overall = round(clarity * 0.3 + specificity * 0.4 + tone * 0.3, 3)

        return {
            "clarity": round(clarity, 3),
            "specificity": round(specificity, 3),
            "tone": round(tone, 3),
            "overall": overall,
        }

    # ──────────────────────────────────────────────
    #  Full Stage 2 Analysis
    # ──────────────────────────────────────────────

    def analyze_critique(self, text: str) -> dict:
        """
        Run full Stage 2 analysis on a single critique.

        Returns dict with: sentiment, intent, quality scores.
        (Clustering is done at the work level across multiple critiques.)
        """
        sentiment = self.classify_sentiment(text)
        intent = self.recognize_intent(text)
        quality = self.score_quality(text, sentiment)

        # Backward-compatible tone_score and actionability_score
        tone_score = quality["tone"]
        actionability_score = quality["specificity"]

        return {
            "sentiment": sentiment,
            "intent": intent,
            "quality": quality,
            "tone_score": tone_score,
            "actionability_score": actionability_score,
        }

    def analyze_and_store(self, critique_id: int) -> dict:
        """Analyze a critique from the database and store full results."""
        try:
            raw_text = self.db.get_critique_text(critique_id)
            if not raw_text:
                print(f"Error: Critique with ID {critique_id} not found")
                return None

            # Run full analysis
            results = self.analyze_critique(raw_text)

            # Store in database
            self.db.update_critique_analysis(
                critique_id,
                tone_score=results["tone_score"],
                actionability_score=results["actionability_score"],
                sentiment_polarity=results["sentiment"]["polarity"],
                sentiment_intensity=results["sentiment"]["intensity"],
                intent=results["intent"]["primary_intent"],
                quality_clarity=results["quality"]["clarity"],
                quality_specificity=results["quality"]["specificity"],
                quality_overall=results["quality"]["overall"],
            )

            return results

        except Exception as e:
            print(f"Error analyzing critique: {e}")
            return None