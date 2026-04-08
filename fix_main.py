import re

with open("main.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace analyzer import
code = re.sub(r'from analyzer import BERTAnalyzer', 'from analyzer import Stage2Analyzer', code)
code = re.sub(r'analyzer = BERTAnalyzer\(db=db\)', 'analyzer = Stage2Analyzer(db=db)', code)

# Clean up any leftover old code structures before inserting the new one
code = re.sub(r'# ──.*def process_critique_pipeline.*?return \{.*?\n    \}\n', '', code, flags=re.DOTALL)

# Insert the helper code right before API endpoints
helper_code = """
# ──────────────────────────────────────────────
#  Full Pipeline Orchestration Helpers
# ──────────────────────────────────────────────

def process_critique_1_and_2a(critique_id: int, raw_text: str, aspect: str = "") -> dict:
    \"\"\"
    Run Stage 1 and Stage 2A on a single critique.
    \"\"\"
    # ── Stage 1: Preprocessing ──
    preprocessed = preprocessor.preprocess(raw_text)

    # Store preprocessing results
    db.update_critique_preprocessing(
        critique_id,
        cleaned_text=preprocessed["cleaned_text"],
        is_meaningful=preprocessed["is_meaningful"],
    )

    if not preprocessed["is_meaningful"]:
        return {
            "critique_id": critique_id,
            "is_meaningful": False,
            "message": "Feedback filtered as non-actionable",
        }

    # ── Stage 2A: Semantic Analysis ──
    analysis_2a = analyzer.analyze_critique_2a(
        preprocessed["cleaned_text"], 
        preprocessed["embedding"],
        preprocessed.get("length", 0)
    )
    
    # Store Stage 2A results
    db.update_critique_analysis(
        critique_id,
        tone_score=analysis_2a["quality_scores"]["tone"],
        actionability_score=analysis_2a["quality_scores"]["specificity"],
        sentiment_polarity=analysis_2a["sentiment"]["polarity"],
        sentiment_intensity=analysis_2a["sentiment"]["intensity"],
        intent=analysis_2a["intent"]["primary_intent"],
        quality_clarity=analysis_2a["quality_scores"]["clarity"],
        quality_specificity=analysis_2a["quality_scores"]["specificity"],
        quality_overall=analysis_2a["quality_scores"]["overall"]
    )

    return {
        "critique_id": critique_id,
        "is_meaningful": True,
        "preprocessing": {"token_count": preprocessed.get("length", 0)},
        "analysis": analysis_2a
    }

def process_batch_2b_3(critiques_data):
    \"\"\"
    Run Stage 2B (clustering) and Stage 3 (generative synthesis per cluster).
    Returns list of enhanced clusters.
    \"\"\"
    if not critiques_data:
        return []
    
    # 1. Collect embeddings (Stage 2B Prep)
    embeddings_map = {}
    valid_cids = []
    for c in critiques_data:
        if c.get("is_meaningful"):
            emb = preprocessor.vectorize(c.get("cleaned_text") or c.get("raw_text", ""))
            if emb is not None and len(emb) > 0:
                embeddings_map[c["id"]] = emb
                valid_cids.append(c["id"])
            
    # 2. Stage 2B: Batch Cluster
    clustering = analyzer.batch_cluster_2b(embeddings_map)
    for cid in valid_cids:
        db.update_critique_cluster(cid, clustering.get(cid, 0))
        
    # Group critiques
    clusters = {}
    for c in critiques_data:
        if not c.get("is_meaningful"): continue
        cid = c["id"]
        c_id = clustering.get(cid, 0)
        if c_id not in clusters:
            clusters[c_id] = {"cluster_id": c_id, "critiques": [], "intent_counts": {}, "sent_counts": {}, "quality_sum": 0.0, "total": 0}
        
        clusters[c_id]["critiques"].append(c["raw_text"])
        clusters[c_id]["quality_sum"] += c.get("quality_overall", 0.5)
        clusters[c_id]["total"] += 1
        
        intent = c.get("intent", "observation")
        clusters[c_id]["intent_counts"][intent] = clusters[c_id]["intent_counts"].get(intent, 0) + 1
        
        sent = c.get("sentiment_polarity", "neutral")
        clusters[c_id]["sent_counts"][sent] = clusters[c_id]["sent_counts"].get(sent, 0) + 1

    # 3. Stage 3: Enhance each cluster
    final_clusters = []
    for c_id, c_data in clusters.items():
        total = max(1, c_data["total"])
        dominant_intent = max(c_data["intent_counts"], key=c_data["intent_counts"].get)
        dominant_sent = max(c_data["sent_counts"], key=c_data["sent_counts"].get)
        
        cluster_payload = {
            "cluster_id": c_id,
            "critiques": c_data["critiques"],
            "avg_sentiment": dominant_sent,
            "dominant_intent": dominant_intent,
            "avg_quality": c_data["quality_sum"] / total
        }
        
        # Generative Enhancement
        enhancements = enhancer.process_cluster(cluster_payload)
        cluster_payload.update(enhancements)
        final_clusters.append(cluster_payload)

    return final_clusters

# ──────────────────────────────────────────────
#  API Endpoints
"""
code = re.sub(r'# ──────────────────────────────────────────────\n#  API Endpoints', helper_code, code)

# Update /api/works/{work_id}/critiques
code = re.sub(r'@app\.post\("/api/works/{work_id}/critiques"\).*?def add_critique[^\n]*\n.*?return result\n', 
'''@app.post("/api/works/{work_id}/critiques")
async def add_critique(work_id: int, critique: Critique):
    """
    Add a critique and process it through Stage 1 & 2A (per-critique analysis).
    """
    critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
    if not critique_id:
        raise HTTPException(status_code=500, detail="Failed to add critique")

    result = process_critique_1_and_2a(critique_id, critique.raw_text, critique.aspect)
    return result
''', code, flags=re.DOTALL)

# Update /api/process-feedback
code = re.sub(r'@app\.post\("/api/process-feedback"\).*?def process_feedback.*?(?=\n\n#|$)', 
'''@app.post("/api/process-feedback")
async def process_feedback(data: WorkWithCritiques):
    """
    Process a work and its critiques through the full batch pipeline.
    Stage 1 → Stage 2A → Stage 2B → Stage 3 → Stage 4.
    """
    work_id = collector.add_work(data.work.user_id, data.work.content, data.work.type)
    if not work_id:
        raise HTTPException(status_code=500, detail="Failed to create work")

    # Stages 1 & 2A
    for critique in data.critiques:
        critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
        if critique_id:
            process_critique_1_and_2a(critique_id, critique.raw_text, critique.aspect)

    # Stages 2B & 3
    critiques_data = db.get_all_critique_details_for_work(work_id)
    clusters = process_batch_2b_3(critiques_data)
    
    # Stage 4
    report = synthesizer.synthesize_clusters(clusters)

    return {"work_id": work_id, "report": report}''', code, flags=re.DOTALL)

# Update /api/works/{work_id}/report
code = re.sub(r'@app\.get\("/api/works/{work_id}/report"\).*?def generate_report[^\n]*\n.*?return \{.*?\}', 
'''@app.get("/api/works/{work_id}/report")
async def generate_report(work_id: int):
    """
    Generate the structured report on demand (runs 2B, 3, and 4 on saved data).
    """
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    critiques_data = db.get_all_critique_details_for_work(work_id)
    if not critiques_data:
        raise HTTPException(status_code=404, detail="No critiques found to report")

    # Re-run batch clustering and enhancement
    clusters = process_batch_2b_3(critiques_data)
    
    # Synthesize
    report = synthesizer.synthesize_clusters(clusters)

    return {"work_id": work_id, "work_content": work[2], "work_type": work[3], "report": report}''', code, flags=re.DOTALL)


with open("main.py", "w", encoding="utf-8") as f:
    f.write(code)
    
print("Successfully modified main.py")
