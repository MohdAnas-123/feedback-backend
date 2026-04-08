"""
Stage 3: Generative Synthesis (Cluster-Level)

Paper reference: "The enhancer reformulates critical or vague critiques
into actionable and supportive recommendations while preserving the
original meaning."

Refactored to operate at the CLUSTER level to minimize LLM calls.
"""
import json

class GPTEnhancer:
    """
    Stage 3 of the Critique Connect pipeline.
    Reformulates and enhances a CLUSTER of semantically similar critiques at once.
    """

    def __init__(self, gemini_agent=None, db=None):
        """
        Initialize the enhancer.
        Args:
            gemini_agent: Gemini instance from Agent.google for generation.
            db: Database instance (optional, kept for signature backwards compatibility).
        """
        self.agent = gemini_agent

    def process_cluster(self, cluster_data: dict) -> dict:
        """
        Processes an entire cluster of critiques in a single Gemini call.
        Input: {
            "cluster_id": int, 
            "critiques": [...raw texts...], 
            "avg_sentiment": "positive",
            "dominant_intent": "suggestion", 
            "avg_quality": 0.6
        }
        
        Returns a dict:
        {
            "cluster_summary": str,
            "recommendations": [{"text": str, "rationale": str}],
            "balanced_perspective": str
        }
        """
        if not self.agent:
            print("Warning: Gemini agent not configured. Skipping enhancement.")
            return self._fallback_enhancement(cluster_data)

        if not cluster_data.get("critiques"):
            return self._fallback_enhancement(cluster_data)

        # 1. Format the input for the agent
        critiques_list = "\n".join([f"- {text}" for text in cluster_data["critiques"]])
        
        prompt = f"""
        You are an expert creative reviewer and feedback coach. 
        You have been given a cluster of related critiques from multiple peers.
        
        Cluster context:
        - Sentiment: {cluster_data.get('avg_sentiment', 'neutral')}
        - Primary Intent: {cluster_data.get('dominant_intent', 'observation')}
        
        Critiques in this cluster:
        {critiques_list}
        
        Your task is to synthesize this cluster into actionable, balanced feedback.
        Perform the following 3 steps:
        1. Summarize the core theme of these critiques.
        2. Generate 1-2 actionable recommendations based on them.
        3. Provide a 'balanced perspective' (a constructive, encouraging rephrasing of the core issue).
        
        Output MUST be valid JSON matching this schema exactly:
        {{
            "cluster_summary": "<summary string>",
            "recommendations": [
                {{"text": "<actionable recommendation>", "rationale": "<why this helps>"}}
            ],
            "balanced_perspective": "<constructive phrasing>"
        }}
        """

        try:
            # 2. Call Gemini ONCE per cluster
            response = self.agent.generate(prompt)
            text = response.text.strip()
            
            # Clean possible markdown formatting
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
                
            result = json.loads(text.strip())
            return result
        except Exception as e:
            print(f"Gemini cluster enhancement error: {e}")
            return self._fallback_enhancement(cluster_data)

    def _fallback_enhancement(self, cluster_data: dict) -> dict:
        """Fallback if Gemini fails."""
        critiques = cluster_data.get("critiques", [])
        if not critiques:
            summary = "No critiques provided."
        else:
            summary = f"Cluster containing {len(critiques)} critiques regarding a shared theme."
            
        return {
            "cluster_summary": summary,
            "recommendations": [
                {"text": "Consider reviewing this aspect of your work based on peer feedback.", "rationale": "General improvement."}
            ],
            "balanced_perspective": "Keep iterating on this area to improve the overall quality."
        }
