"""
Stage 3 + 4: Synthesis and Final Report Generation

Paper reference:
Stage 3: "Synthesize... cluster summarisation, recommendation for action, balanced perspective"
Stage 4: "An aggregated multi-dimensional feedback report is compiled...
  - Quick digest + sentiment overview
  - Strengths and weaknesses sections
  - Recommendations for action, ranked by impact and occurrence"

Uses Gemini for generative synthesis. BERT analysis feeds into structured report.
"""

import json
from database import Database


class Synthesizer:
    """
    Stages 3-4 of the Critique Connect pipeline.
    Synthesizes multiple analysed critiques into a structured final report.

    Hybrid pipeline: receives BERT-analysed data → generates structured output via Gemini.
    """

    def __init__(self, gemini_agent=None, db=None):
        """
        Initialize the synthesizer.

        Args:
            gemini_agent: Gemini instance for generative synthesis.
            db: Shared Database instance.
        """
        self.agent = gemini_agent
        self.db = db or Database()

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini and return text response."""
        if self.agent is None:
            return None
        try:
            response = self.agent.generate(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini synthesis error: {e}")
            return None

    # ──────────────────────────────────────────────
    #  Structured Synthesis (Paper Stage 3)
    # ──────────────────────────────────────────────

    def synthesize_critiques(self, critiques_data: list) -> dict:
        """
        Synthesize multiple critiques into a structured report.

        Args:
            critiques_data: List of dicts with keys:
                raw_text, enhanced_text, sentiment_polarity, sentiment_intensity,
                intent, quality_overall, aspect, cluster_id

        Returns:
            Structured report dict matching Paper Stage 4 output.
        """
        if not critiques_data:
            return self._empty_report()

        # ── Build context for Gemini ──
        formatted_critiques = []
        for i, c in enumerate(critiques_data, 1):
            entry = f"Critique {i}:"
            entry += f"\n  Aspect: {c.get('aspect', 'general')}"
            entry += f"\n  Raw: {c.get('raw_text', '')}"
            if c.get('enhanced_text'):
                entry += f"\n  Enhanced: {c['enhanced_text']}"
            entry += f"\n  Sentiment: {c.get('sentiment_polarity', 'unknown')} (intensity: {c.get('sentiment_intensity', 0)})"
            entry += f"\n  Intent: {c.get('intent', 'unknown')}"
            entry += f"\n  Quality: {c.get('quality_overall', 0)}"
            formatted_critiques.append(entry)

        critiques_text = "\n\n".join(formatted_critiques)

        # ── Aggregate sentiment stats ──
        sentiment_stats = self._compute_sentiment_stats(critiques_data)

        prompt = f"""You are an expert feedback synthesis engine for creative work review.

You have been given {len(critiques_data)} analysed critiques. Generate a structured synthesis report.

ANALYSED CRITIQUES:
{critiques_text}

SENTIMENT DISTRIBUTION:
- Positive: {sentiment_stats['positive_pct']}%
- Neutral: {sentiment_stats['neutral_pct']}%
- Negative: {sentiment_stats['negative_pct']}%

Generate a JSON response with this EXACT structure:
{{
    "digest": "<A concise 2-3 sentence overall summary of all feedback>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "recommendations": [
        {{"text": "<specific actionable recommendation>", "impact": "high/medium/low", "frequency": <number of critiques mentioning this>}},
        ...
    ]
}}

Rules:
- Extract real patterns from the critiques, don't invent issues
- Rank recommendations by impact (high first) then frequency
- Strengths come from positive critiques, weaknesses from negative/constructive ones
- Be specific and actionable
- Output valid JSON only, no markdown code blocks"""

        result = self._call_gemini(prompt)

        if result:
            try:
                # Clean potential markdown formatting
                cleaned = result.strip().strip("`").strip()
                if cleaned.startswith("json"):
                    cleaned = "\n".join(cleaned.split("\n")[1:]).strip()
                parsed = json.loads(cleaned)

                # Merge with computed stats
                parsed["sentiment_overview"] = {
                    "positive": sentiment_stats["positive_pct"],
                    "neutral": sentiment_stats["neutral_pct"],
                    "negative": sentiment_stats["negative_pct"],
                }
                parsed["critique_count"] = len(critiques_data)
                parsed["average_quality"] = sentiment_stats["avg_quality"]

                return parsed

            except json.JSONDecodeError as e:
                print(f"Failed to parse Gemini synthesis JSON: {e}")
                # Fall through to rule-based

        # Fallback: rule-based synthesis
        return self._rule_based_synthesis(critiques_data, sentiment_stats)

    # ──────────────────────────────────────────────
    #  Sentiment Statistics
    # ──────────────────────────────────────────────

    def _compute_sentiment_stats(self, critiques_data: list) -> dict:
        """Compute sentiment distribution and average quality from analysed critiques."""
        total = len(critiques_data)
        if total == 0:
            return {"positive_pct": 0, "neutral_pct": 0, "negative_pct": 0, "avg_quality": 0}

        positive = sum(1 for c in critiques_data if c.get("sentiment_polarity") == "positive")
        neutral = sum(1 for c in critiques_data if c.get("sentiment_polarity") == "neutral")
        negative = sum(1 for c in critiques_data if c.get("sentiment_polarity") == "negative")

        qualities = [c.get("quality_overall", 0) for c in critiques_data if c.get("quality_overall")]
        avg_quality = round(sum(qualities) / len(qualities), 2) if qualities else 0

        return {
            "positive_pct": round(positive / total * 100),
            "neutral_pct": round(neutral / total * 100),
            "negative_pct": round(negative / total * 100),
            "avg_quality": avg_quality,
        }

    # ──────────────────────────────────────────────
    #  Work-level synthesis (from DB)
    # ──────────────────────────────────────────────

    def synthesize_work_critiques(self, work_id: int) -> dict:
        """
        Generate a full structured report for all critiques of a work.
        Fetches analysed critiques from DB and synthesizes.
        """
        critiques_data = self.db.get_enhanced_critiques_for_work(work_id)

        if not critiques_data:
            return self._empty_report()

        return self.synthesize_critiques(critiques_data)

    # ──────────────────────────────────────────────
    #  Fallback: Rule-based synthesis
    # ──────────────────────────────────────────────

    def _rule_based_synthesis(self, critiques_data: list, sentiment_stats: dict) -> dict:
        """Fallback synthesis when Gemini is unavailable."""
        total = len(critiques_data)

        # Extract common themes from aspects
        aspects = {}
        for c in critiques_data:
            aspect = c.get("aspect", "general")
            aspects[aspect] = aspects.get(aspect, 0) + 1

        # Build strengths from positive critiques
        strengths = []
        weaknesses = []
        for c in critiques_data:
            text = c.get("enhanced_text") or c.get("raw_text", "")
            if c.get("sentiment_polarity") == "positive":
                strengths.append(text)
            elif c.get("sentiment_polarity") == "negative":
                weaknesses.append(text)

        # Build recommendations from suggestions/criticisms
        recommendations = []
        for c in critiques_data:
            if c.get("intent") in ("suggestion", "criticism"):
                text = c.get("enhanced_text") or c.get("raw_text", "")
                recommendations.append({
                    "text": text,
                    "impact": "high" if c.get("quality_overall", 0) > 0.6 else "medium",
                    "frequency": 1,
                })

        # Deduplicate and sort recommendations by impact
        impact_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: impact_order.get(r["impact"], 2))

        digest = (
            f"Based on {total} critiques, the overall sentiment is "
            f"{sentiment_stats['positive_pct']}% positive, "
            f"{sentiment_stats['neutral_pct']}% neutral, and "
            f"{sentiment_stats['negative_pct']}% negative. "
            f"Key areas of focus include: {', '.join(list(aspects.keys())[:3])}."
        )

        return {
            "digest": digest,
            "sentiment_overview": {
                "positive": sentiment_stats["positive_pct"],
                "neutral": sentiment_stats["neutral_pct"],
                "negative": sentiment_stats["negative_pct"],
            },
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "recommendations": recommendations[:5],
            "critique_count": total,
            "average_quality": sentiment_stats["avg_quality"],
        }

    def _empty_report(self) -> dict:
        """Return an empty report structure."""
        return {
            "digest": "No critiques available for synthesis.",
            "sentiment_overview": {"positive": 0, "neutral": 0, "negative": 0},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "critique_count": 0,
            "average_quality": 0,
        }
