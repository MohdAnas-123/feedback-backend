"""
Stage 3: Developing Practical Recommendations — Generative Synthesis

Paper reference: "A generative synthesis model... serves three main purposes:
1. Cluster summarisation  2. Recommendation for action from feedback
3. Balanced perspective"

Uses Gemini (Google Generative AI) as the generation backbone.
GPT-2 removed per user decision — Gemini handles all generation tasks.
"""

import json
from database import Database


class GPTEnhancer:
    """
    Stage 3 of the Critique Connect pipeline.
    Enhances raw feedback into constructive, actionable suggestions using Gemini.

    Hybrid architecture: BERT analyzes (Stage 2) → Gemini generates (Stage 3).
    """

    def __init__(self, gemini_agent=None, db=None):
        """
        Initialize the enhancer.

        Args:
            gemini_agent: Gemini instance from Agent.google for generation.
            db: Shared Database instance.
        """
        self.agent = gemini_agent
        self.db = db or Database()

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini and return text response. Returns None on failure."""
        if self.agent is None:
            print("Warning: Gemini agent not available. Using rule-based fallback.")
            return None
        try:
            response = self.agent.generate(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return None

    # ──────────────────────────────────────────────
    #  Paper Stage 3 — Sub-task 1: Cluster Summarisation
    # ──────────────────────────────────────────────

    def summarize_cluster(self, cluster_critiques: list) -> str:
        """
        Produce a concise summary for a thematic cluster of critiques.

        Paper: "For each thematic cluster, the generative model produces shorter
        summarisations that account for majority feedback without redundancy
        yet maintain opinion diversity."
        """
        formatted = "\n".join(f"- {c}" for c in cluster_critiques)

        prompt = f"""You are an expert feedback analyst for creative work review.

Summarize the following group of related critiques into a single concise paragraph.
Capture the majority opinion while preserving diverse viewpoints. Remove redundancy.

Critiques:
{formatted}

Write only the summary paragraph, nothing else."""

        result = self._call_gemini(prompt)
        if result:
            return result

        # Fallback: concatenate unique points
        return " ".join(set(cluster_critiques))

    # ──────────────────────────────────────────────
    #  Paper Stage 3 — Sub-task 2: Actionable Recommendations
    # ──────────────────────────────────────────────

    def generate_recommendations(self, critique_text: str, aspect: str = "") -> str:
        """
        Transform feedback analysis into concrete, actionable recommendations.

        Paper: "The model translates the feedback into concrete block recommendations
        or directions in terms of suggested best practices/next steps."
        """
        aspect_context = f" about '{aspect}'" if aspect else ""

        prompt = f"""You are an expert feedback coach for creative professionals.

Transform the following raw critique{aspect_context} into clear, actionable recommendations.
Make it constructive, specific, and immediately actionable. Maintain the core message.

Raw critique: "{critique_text}"

Provide the enhanced feedback as a single paragraph with concrete suggestions. Nothing else."""

        result = self._call_gemini(prompt)
        if result:
            return result

        # Fallback: rule-based enhancement
        return self._rule_based_enhancement(critique_text)

    # ──────────────────────────────────────────────
    #  Paper Stage 3 — Sub-task 3: Balanced Perspective
    # ──────────────────────────────────────────────

    def balance_perspective(self, enhanced_text: str, original_text: str) -> str:
        """
        Ensure the enhanced feedback balances positive and negative aspects.

        Paper: "The model makes sure that each review has a balance between
        positive feedback and negative criticism ratings."
        """
        prompt = f"""You are an expert feedback editor ensuring balanced critique.

The following enhanced feedback may lean too negative or too positive.
Rewrite it to maintain a balance: acknowledge strengths before addressing weaknesses.
Keep it constructive and professional.

Original feedback: "{original_text}"
Enhanced feedback: "{enhanced_text}"

Write only the balanced version, nothing else."""

        result = self._call_gemini(prompt)
        return result if result else enhanced_text

    # ──────────────────────────────────────────────
    #  Full Enhancement Pipeline
    # ──────────────────────────────────────────────

    def enhance_critique(self, critique_text: str, aspect: str = "") -> str:
        """
        Full Stage 3 enhancement: generate actionable recommendations
        with balanced perspective.
        """
        # Step 1: Generate actionable recommendations
        enhanced = self.generate_recommendations(critique_text, aspect)

        # Step 2: Balance perspective
        balanced = self.balance_perspective(enhanced, critique_text)

        return balanced

    def enhance_and_store(self, critique_id: int) -> str:
        """Enhance a critique from the database and store the result."""
        try:
            raw_text = self.db.get_critique_text(critique_id)
            if not raw_text:
                print(f"Error: Critique with ID {critique_id} not found")
                return None

            # Get aspect for context (if available)
            self.db.cursor.execute(
                "SELECT aspect FROM critiques WHERE id = ?", (critique_id,)
            )
            row = self.db.cursor.fetchone()
            aspect = row[0] if row else ""

            # Enhance
            enhanced_text = self.enhance_critique(raw_text, aspect)

            # Store
            self.db.update_critique_enhanced_text(critique_id, enhanced_text)

            return enhanced_text

        except Exception as e:
            print(f"Error enhancing critique: {e}")
            return None

    # ──────────────────────────────────────────────
    #  Fallback: Rule-based enhancement (no Gemini)
    # ──────────────────────────────────────────────

    def _rule_based_enhancement(self, critique_text: str) -> str:
        """Fallback method when Gemini is unavailable."""
        enhancements = {
            "bad": "could be improved",
            "terrible": "needs significant improvement",
            "awful": "requires substantial changes",
            "horrible": "needs major revision",
            "worst": "could be better",
            "hate": "prefer a different approach",
            "stupid": "could be more thoughtful",
            "useless": "could be more useful",
            "waste": "could be more efficient",
            "pointless": "could serve a better purpose",
            "ugly": "could benefit from visual refinement",
            "boring": "could be more engaging",
            "confusing": "could be clearer",
        }

        enhanced_text = critique_text
        for negative, positive in enhancements.items():
            # Case-insensitive replacement
            import re
            enhanced_text = re.sub(
                rf"\b{negative}\b", positive, enhanced_text, flags=re.IGNORECASE
            )

        return enhanced_text
