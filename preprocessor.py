"""
Stage 1: Input and Preparation — Raw Feedback → Preprocessing

Paper reference: "The pipeline starts with data normalisation, converting feedback to a
standardised text format... sentential and word segmentation... noise removal such as
stop words, lemmatization... context filters catch garbage or empty comments...
Vectorise the pre-processed feedback for semantic analysis with embedding models."
"""

import re
import string
import numpy as np


# ──────────────────────────────────────────────
#  Lightweight preprocessing (no heavy dependencies)
#  NLTK / sentence-transformers loaded lazily
# ──────────────────────────────────────────────

class FeedbackPreprocessor:
    """
    Stage 1 of the Critique Connect pipeline.
    Cleans, filters, and vectorizes raw peer feedback for downstream analysis.
    """

    # Low-value phrases that pass no actionable information (context filter)
    GARBAGE_PHRASES = {
        "good job", "nice work", "well done", "great", "awesome", "cool",
        "nice", "ok", "okay", "fine", "good", "bad", "not bad", "looks good",
        "looks great", "love it", "hate it", "like it", "interesting",
        "amazing", "perfect", "terrible", "wow", "lol", "idk",
        "no comment", "no comments", "nothing to say", "n/a", "na",
    }

    # Common English stop words (subset — avoids mandatory NLTK download)
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "but", "or", "not", "so", "if", "then", "than",
        "very", "just", "also", "too", "more", "most", "some", "any",
        "all", "each", "every", "both", "few", "many", "much",
        "about", "up", "out", "into", "over", "after", "before",
    }

    def __init__(self):
        self._sbert_model = None  # lazy-loaded

    # ── Sub-step 1: Normalization ─────────────────

    def normalize(self, text: str) -> str:
        """
        Data normalisation: convert to standardised text format.
        - Lowercase
        - Strip HTML tags
        - Regularize whitespace
        - Remove redundant symbols
        """
        if not text:
            return ""

        text = text.lower().strip()

        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # Normalize unicode quotes and dashes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2014", " - ").replace("\u2013", " - ")

        # Remove excessive punctuation repetition (e.g., "!!!" → "!")
        text = re.sub(r"([!?.]){2,}", r"\1", text)

        # Regularize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── Sub-step 2: Tokenization ─────────────────

    def tokenize(self, text: str) -> list:
        """
        Sentence and word segmentation for fine-grained linguistic analysis.
        Returns list of word tokens.
        """
        if not text:
            return []

        # Simple word tokenization: split on whitespace and punctuation boundaries
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
        return tokens

    # ── Sub-step 3: Noise Removal ────────────────

    def remove_noise(self, text: str) -> str:
        """
        Stop word removal and vocabulary normalization.
        Keeps domain-relevant words while removing common noise.
        """
        tokens = self.tokenize(text)

        # Remove stop words and single-character tokens
        filtered = [
            token for token in tokens
            if token not in self.STOP_WORDS and len(token) > 1
        ]

        return " ".join(filtered)

    # ── Sub-step 4: Context Filtering ────────────

    def filter_context(self, text: str) -> bool:
        """
        Context filter: detect garbage or empty comments.
        Returns True if the feedback is meaningful, False if it's low-value.

        Paper: "Context filters catch garbage or empty comments — 'good job',
        'nice work' — that don't really accomplish anything."
        """
        if not text:
            return False

        normalized = text.lower().strip()

        # Remove punctuation for comparison
        clean = normalized.translate(str.maketrans("", "", string.punctuation)).strip()

        # Check against known garbage phrases
        if clean in self.GARBAGE_PHRASES:
            return False

        # Too short to be meaningful (less than 3 words after cleaning)
        words = clean.split()
        if len(words) < 3:
            return False

        # Check if it's just repeated characters or nonsense
        unique_chars = set(clean.replace(" ", ""))
        if len(unique_chars) < 3:
            return False

        return True

    # ── Sub-step 5: Embedding Vectorization ──────

    def vectorize(self, text: str) -> list:
        """
        Generate SBERT embeddings for semantic analysis and clustering.
        Uses sentence-transformers for domain-aware embeddings.
        Lazy-loads the model on first call.
        """
        try:
            if self._sbert_model is None:
                from sentence_transformers import SentenceTransformer
                self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Loaded SBERT model: all-MiniLM-L6-v2")

            embedding = self._sbert_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except ImportError:
            print("Warning: sentence-transformers not installed. Using fallback embeddings.")
            return self._fallback_embeddings(text)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return self._fallback_embeddings(text)

    def _fallback_embeddings(self, text: str) -> list:
        """Simple TF-based fallback when sentence-transformers is unavailable."""
        tokens = self.tokenize(text)
        # Create a simple bag-of-words hash-based embedding (64 dims)
        embedding = np.zeros(64)
        for token in tokens:
            idx = hash(token) % 64
            embedding[idx] += 1
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    # ── Full Pipeline ────────────────────────────

    def preprocess(self, text: str) -> dict:
        """
        Full Stage 1 preprocessing pipeline.

        Returns:
            dict with keys:
                - original_text: the raw input
                - cleaned_text: normalized and noise-removed text
                - tokens: word tokens
                - embedding: SBERT vector (list of floats)
                - is_meaningful: bool — whether the feedback passes context filtering
        """
        # Step 1: Normalize
        normalized = self.normalize(text)

        # Step 4: Context filter (on normalized text, before noise removal)
        is_meaningful = self.filter_context(normalized)

        # Step 3: Noise removal
        cleaned = self.remove_noise(normalized)

        # Step 2: Tokenize (the cleaned version)
        tokens = self.tokenize(cleaned)

        # Step 5: Vectorize (use normalized text for better embeddings)
        embedding = self.vectorize(normalized) if is_meaningful else []

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "embedding": embedding,
            "is_meaningful": is_meaningful,
        }
