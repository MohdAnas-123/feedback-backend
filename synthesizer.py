"""
Stage 4: Synthesizer

Paper reference: "The synthesizer consolidates the outputs from
all previous stages to produce a final, structured report."

Refactored to consume CLUSTERED data and enforce deterministic
aggregation prior to LLM generation.
"""
import json

class Synthesizer:
    """
    Stage 4 of the Critique Connect pipeline.
    Produces the final structured report from clustered and enhanced data.
    """

    def __init__(self, gemini_agent=None, db=None):
        self.agent = gemini_agent

    def synthesize_clusters(self, clusters: list) -> dict:
        """
        Consumes clustered + enhanced data to generate the final report.
        
        Input format:
        clusters = [
            {
                "cluster_id": int,
                "critiques": [...],
                "avg_sentiment": "...",
                "dominant_intent": "...",
                "avg_quality": float,
                "cluster_summary": "...",
                "recommendations": [...],
                "balanced_perspective": "..."
            }
        ]
        """
        if not clusters:
            return self._empty_report()
            
        # 1. Deterministic Aggregation
        total_critiques = 0
        quality_sum = 0.0
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        # Prepare deterministic recommendations with calculated impact/frequency
        deterministic_recs = []
        cluster_digests = []
        
        for cluster in clusters:
            cluster_critiques_count = len(cluster["critiques"])
            total_critiques += cluster_critiques_count
            quality_sum += (cluster.get("avg_quality", 0.0) * cluster_critiques_count)
            
            # Simple tally for sentiment based on the cluster's avg sentiment
            # In a real app we might sum individual critiques, but for brevity cluster avg is fine
            sent = cluster.get("avg_sentiment", "neutral")
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + cluster_critiques_count
            
            cluster_digests.append(f"Theme {cluster['cluster_id']}: {cluster.get('cluster_summary', '')}")
            
            # Derive Impact = Base Quality * (High if negative, low if praise)
            base_impact = "medium"
            if cluster.get("avg_quality", 0.5) > 0.6:
                if sent == "negative":
                    base_impact = "high" # High quality negative critique -> High impact to fix
                elif sent == "positive":
                    base_impact = "low"  # High quality positive critique -> low impact to fix
            
            # Pass through the enhanced recommendations
            for rec in cluster.get("recommendations", []):
                deterministic_recs.append({
                    "text": rec.get("text", ""),
                    "impact": base_impact,
                    "frequency": cluster_critiques_count
                })

        avg_quality = round(quality_sum / max(1, total_critiques), 2)
        
        # Format sentiment percentages
        sentiment_overview = {
            "positive": f"{round((sentiment_counts['positive'] / total_critiques) * 100)}%",
            "neutral": f"{round((sentiment_counts['neutral'] / total_critiques) * 100)}%",
            "negative": f"{round((sentiment_counts['negative'] / total_critiques) * 100)}%"
        }
        
        # Sort recommendations by frequency and impact deterministically
        impact_weights = {"high": 3, "medium": 2, "low": 1}
        deterministic_recs.sort(key=lambda x: (x["frequency"], impact_weights.get(x["impact"], 0)), reverse=True)
        
        # 2. Generative Formatting (Gemini)
        # We ask Gemini to generate the digest, strengths, and weaknesses from the cluster summaries.
        # We pass back the deterministic numbers to maintain strict truth.
        report = {
            "sentiment_overview": sentiment_overview,
            "critique_count": total_critiques,
            "average_quality": avg_quality,
            "recommendations": deterministic_recs
        }

        if not self.agent:
            report["digest"] = "Analyzed feedback locally."
            report["strengths"] = ["Review clusters for praise."]
            report["weaknesses"] = ["Review clusters for criticism."]
            return report

        clusters_text = "\n".join(cluster_digests)
        prompt = f"""
        Based on these thematic summaries of peer feedback:
        {clusters_text}
        
        Write a JSON response with:
        1. "digest": A 2-3 sentence executive summary of the overall feedback.
        2. "strengths": Array of 2-3 major strengths identified in the feedback.
        3. "weaknesses": Array of 2-3 major weaknesses or areas for improvement.
        
        Respond with ONLY valid JSON formatting correctly:
        {{
            "digest": "...",
            "strengths": ["...", "..."],
            "weaknesses": ["...", "..."]
        }}
        """

        try:
            res = self.agent.generate(prompt)
            text = res.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.endswith("```"): text = text[:-3]
            ai_data = json.loads(text.strip())
            
            report["digest"] = ai_data.get("digest", "")
            report["strengths"] = ai_data.get("strengths", [])
            report["weaknesses"] = ai_data.get("weaknesses", [])
        except Exception as e:
            print(f"Synthesis AI formatting error: {e}")
            report["digest"] = "Failed to generate digest."
            report["strengths"] = []
            report["weaknesses"] = []

        return report

    def _empty_report(self):
        return {
            "digest": "No feedback provided yet.",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "sentiment_overview": {"positive": "0%", "neutral": "0%", "negative": "0%"},
            "critique_count": 0,
            "average_quality": 0.0
        }
