"""
Stage 2: Understanding Feedback — Semantic Analysis

Paper reference: "Once pre-processed, the text undergoes semantic analysis...
1. Sentiment Classification  2. Intent Recognition
3. Thematic Clustering       4. Quality Scoring"

Refactored to Stage 2A (per critique) and Stage 2B (batch clustering).
Removed heavy BART model.
"""

import numpy as np
from database import Database

class Stage2Analyzer:
    """
    Stage 2 of the Critique Connect pipeline.
    Multi-dimensional analysis of peer feedback.
    Models are lazy-loaded on first use.
    """

    INTENT_LABELS = ["praise", "suggestion", "criticism", "question", "observation"]
    
    # Anchor texts for intent mapping via embedding similarity
    ANCHORS = {
        "praise": "This is excellent work, I really love the great job you did.",
        "suggestion": "You should consider changing or improving this by doing something else.",
        "criticism": "This is poorly done, it looks bad and doesn't work well.",
        "question": "How did you do this? Could you explain why it is like this?",
        "observation": "I noticed that this part is located here, the color is blue."
    }

    def __init__(self, db=None):
        """Initialize the analyzer."""
        self.db = db or Database()
        self._sentiment_pipeline = None
        self._anchor_embeddings = None

    # ──────────────────────────────────────────────
    #  Lazy model loading
    # ──────────────────────────────────────────────

    @property
    def sentiment_pipeline(self):
        """Lazy-load lightweight sentiment analysis pipeline."""
        if self._sentiment_pipeline is None:
            from transformers import pipeline
            # Used distilbert-sst2 (~250MB) instead of nlptown (~600MB) for memory safety
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("Loaded sentiment model: distilbert-sst2")
        return self._sentiment_pipeline
    
    def _get_anchor_embeddings(self):
        """Lazy-load and compute intent anchor embeddings."""
        if self._anchor_embeddings is None:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer("all-MiniLM-L6-v2")
            self._anchor_embeddings = {}
            for label, text in self.ANCHORS.items():
                self._anchor_embeddings[label] = sbert.encode(text, convert_to_numpy=True)
        return self._anchor_embeddings

    # ──────────────────────────────────────────────
    #  STAGE 2A: Per Critique Analysis
    # ──────────────────────────────────────────────

    def classify_sentiment(self, text: str) -> dict:
        """
        Evaluate feedback for polarity and intensity using DistilBERT SST-2.
        """
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            label = result["label"]  # POSITIVE or NEGATIVE
            score = result["score"]  # confidence

            if label == "POSITIVE":
                polarity = "positive" if score > 0.6 else "neutral"
                intensity = score
            else:
                polarity = "negative" if score > 0.6 else "neutral"
                intensity = score

            return {
                "polarity": polarity,
                "intensity": round(intensity, 3),
                "confidence": round(score, 3)
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {"polarity": "neutral", "intensity": 0.5, "confidence": 0.0}

    def recognize_intent(self, embedding: list) -> dict:
        """
        Distinguish intents via cosine similarity against SBERT anchor embeddings.
        Replaces zero-shot BART to save ~1.5GB of RAM.
        """
        try:
            if not embedding:
                return {"primary_intent": "observation", "confidence": 0.0}
                
            anchors = self._get_anchor_embeddings()
            emb_vec = np.array(embedding)
            norm_emb = np.linalg.norm(emb_vec)
            if norm_emb == 0:
                raise ValueError("Empty embedding vector")
                
            emb_vec = emb_vec / norm_emb
            
            best_intent = "observation"
            best_score = -1.0
            
            intent_scores = {}
            for label, anchor_vec in anchors.items():
                norm_anchor = np.linalg.norm(anchor_vec)
                anchor_vec = anchor_vec / norm_anchor
                sim = np.dot(emb_vec, anchor_vec)
                intent_scores[label] = round(float(sim), 3)
                if sim > best_score:
                    best_score = sim
                    best_intent = label
                    
            return {
                "primary_intent": best_intent,
                "confidence": round(float(best_score), 3),
                "all_intents": intent_scores
            }
        except Exception as e:
            print(f"Intent recognition error: {e}")
            return {"primary_intent": "observation", "confidence": 0.0}

    def score_quality(self, text: str, sentiment: dict, length: int) -> dict:
        """
        Feature-based scoring model evaluating constructiveness.
        """
        # Clarity
        has_punctuation = any(c in text for c in ".!?;:")
        clarity = min(1.0, (length / 25) * 0.5 + (0.3 if has_punctuation else 0.0) + 0.2)

        # Specificity
        specific_indicators = [
            r"\d+", r"(?:example|e\.g\.|such as|like|instead|consider)",
            r"(?:because|since|due to|reason)", r"(?:should|could|would|try|suggest)",
            r"(?:color|font|layout|spacing|contrast|size|margin|align)"
        ]
        import re
        specificity_hits = sum(1 for pattern in specific_indicators if re.search(pattern, text.lower()))
        specificity = min(1.0, specificity_hits / 3.0)

        # Tone
        tone = 0.5
        if sentiment:
            if sentiment["polarity"] == "negative":
                tone = max(0.1, 1.0 - sentiment["intensity"])
            elif sentiment["polarity"] == "positive":
                tone = min(1.0, 0.5 + sentiment["intensity"] * 0.5)

        overall = round(clarity * 0.3 + specificity * 0.4 + tone * 0.3, 3)

        return {
            "clarity": round(clarity, 3),
            "specificity": round(specificity, 3),
            "tone": round(tone, 3),
            "overall": overall,
        }

    def analyze_critique_2a(self, text: str, embedding: list, length: int) -> dict:
        """
        Stage 2A: Run per-critique semantic analysis.
        """
        sentiment = self.classify_sentiment(text)
        intent = self.recognize_intent(embedding)
        quality = self.score_quality(text, sentiment, length)

        return {
            "sentiment": sentiment,
            "intent": intent,
            "quality_scores": quality
        }

    # ──────────────────────────────────────────────
    #  STAGE 2B: Batch Clustering
    # ──────────────────────────────────────────────

    def batch_cluster_2b(self, embeddings_dict: dict) -> dict:
        """
        Stage 2B: Unsupervised clustering using PCA and KMeans.
        Input: {critique_id: embedding_list}
        Output: {critique_id: cluster_id}
        """
        critique_ids = list(embeddings_dict.keys())
        embeddings_list = [embeddings_dict[cid] for cid in critique_ids]
        
        n = len(embeddings_list)
        if n == 0:
            return {}
            
        # Handle edge case requested by spec: single cluster if n < 3
        if n < 3:
            return {cid: 0 for cid in critique_ids}
            
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            X = np.array(embeddings_list)
            
            # Normalize embeddings
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = np.divide(X, norms, out=np.zeros_like(X), where=norms!=0)
            
            # Apply PCA
            n_components = min(10, n)
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X)
            
            # k = min(5, n//2) as per spec
            k = max(1, min(5, n // 2))
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            
            return {cid: int(label) for cid, label in zip(critique_ids, labels)}
            
        except ImportError:
            print("Warning: scikit-learn not installed. Skipping ML clustering.")
            return {cid: 0 for cid in critique_ids}
        except Exception as e:
            print(f"Batch clustering error: {e}")
            return {cid: 0 for cid in critique_ids}