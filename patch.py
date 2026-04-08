import re

with open("main.py", "r", encoding="utf-8") as f:
    text = f.read()

# Normalize line endings to avoid silently failing replacement
text = text.replace("\r\n", "\n")

# 1. Update imports
text = text.replace("from analyzer import BERTAnalyzer", "from analyzer import Stage2Analyzer")
text = text.replace("analyzer = BERTAnalyzer(db=db)", "analyzer = Stage2Analyzer(db=db)")

# 2. Add orchestration helpers (put them below sanitize_for_prompt)
helpers = """
def process_critique_1_and_2a(critique_id: int, raw_text: str, aspect: str = "") -> dict:
    \"\"\"Run Stage 1 and Stage 2A on a single critique.\"\"\"
    preprocessed = preprocessor.preprocess(raw_text)

    db.update_critique_preprocessing(
        critique_id,
        cleaned_text=preprocessed["cleaned_text"],
        is_meaningful=preprocessed["is_meaningful"],
    )

    if not preprocessed["is_meaningful"]:
        return {"critique_id": critique_id, "is_meaningful": False, "message": "Feedback filtered (non-actionable)"}

    analysis_2a = analyzer.analyze_critique_2a(
        preprocessed["cleaned_text"], 
        preprocessed["embedding"],
        preprocessed.get("length", 0)
    )
    
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

    return {"critique_id": critique_id, "is_meaningful": True, "preprocessing": {"token_count": preprocessed.get("length", 0)}, "analysis": analysis_2a}


def process_batch_2b_3(critiques_data):
    \"\"\"Run Stage 2B (clustering) and Stage 3 (generative synthesis per cluster).\"\"\"
    if not critiques_data: return []
    
    embeddings_map = {}
    valid_cids = []
    for c in critiques_data:
        if c.get("is_meaningful"):
            emb = preprocessor.vectorize(c.get("cleaned_text") or c.get("raw_text", ""))
            if emb is not None and len(emb) > 0:
                embeddings_map[c["id"]] = emb
                valid_cids.append(c["id"])
            
    clustering = analyzer.batch_cluster_2b(embeddings_map)
    for cid in valid_cids:
        db.update_critique_cluster(cid, clustering.get(cid, 0))
        
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
        
        enhancements = enhancer.process_cluster(cluster_payload)
        cluster_payload.update(enhancements)
        final_clusters.append(cluster_payload)

    return final_clusters
"""
original_process_pipeline_str = """def process_critique_pipeline(critique_id: int, raw_text: str, aspect: str = "") -> dict:"""
if "def process_critique_1_and_2a" not in text:
    text = text.replace(original_process_pipeline_str, helpers + "\n" + original_process_pipeline_str)


# 3A. add_critique
old_add_critique = """@app.post("/api/works/{work_id}/critiques")
async def add_critique(work_id: int, critique: Critique):
    \"\"\"
    Add a critique and process it through the full pipeline.
    Stage 1 → Stage 2 → Stage 3
    \"\"\"
    # Add critique to database
    critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
    if not critique_id:
        raise HTTPException(status_code=500, detail="Failed to add critique")

    # Run full pipeline
    result = process_critique_pipeline(critique_id, critique.raw_text, critique.aspect)

    return result"""
new_add_critique = """@app.post("/api/works/{work_id}/critiques")
async def add_critique(work_id: int, critique: Critique):
    critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
    if not critique_id:
        raise HTTPException(status_code=500, detail="Failed to add critique")
    result = process_critique_1_and_2a(critique_id, critique.raw_text, critique.aspect)
    return result"""
if old_add_critique in text:
    text = text.replace(old_add_critique, new_add_critique)
else:
    print("FAILED TO MATCH ADD_CRITIQUE")

# 3B. process_feedback
old_process_feedback = """@app.post("/api/process-feedback")
async def process_feedback(data: WorkWithCritiques):
    \"\"\"
    Process a work and its critiques through the full pipeline in one go.
    Runs Stage 1 → Stage 2 → Stage 3 for each critique, then Stage 4 synthesis.
    \"\"\"
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
    }"""
new_process_feedback = """@app.post("/api/process-feedback")
async def process_feedback(data: WorkWithCritiques):
    work_id = collector.add_work(data.work.user_id, data.work.content, data.work.type)
    if not work_id:
        raise HTTPException(status_code=500, detail="Failed to create work")

    for critique in data.critiques:
        critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
        if critique_id:
            process_critique_1_and_2a(critique_id, critique.raw_text, critique.aspect)

    critiques_data = db.get_all_critique_details_for_work(work_id)
    clusters = process_batch_2b_3(critiques_data)
    
    report = synthesizer.synthesize_clusters(clusters)

    return {"work_id": work_id, "report": report}"""
if old_process_feedback in text:
    text = text.replace(old_process_feedback, new_process_feedback)
else:
    print("FAILED TO MATCH PROCESS_FEEDBACK")

# 3C. generate_report + synthesize_work_critiques
old_gen_report_pattern = r'@app\.get\("/api/works/\{work_id\}/report"\).*?def generate_report\(work_id: int\):.*?return \{\s*"work_id": work_id,\s*"work_content": work\[2\],\s*"work_type": work\[3\],\s*"report": report,\s*\}'
new_gen_report = """@app.get("/api/works/{work_id}/report")
async def generate_report(work_id: int):
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    critiques_data = db.get_all_critique_details_for_work(work_id)
    if not critiques_data:
        raise HTTPException(status_code=404, detail="No critiques found to report")

    clusters = process_batch_2b_3(critiques_data)
    report = synthesizer.synthesize_clusters(clusters)

    return {"work_id": work_id, "work_content": work[2], "work_type": work[3], "report": report}"""
text = re.sub(old_gen_report_pattern, new_gen_report, text, flags=re.DOTALL)

old_synth_crit = """@app.get("/api/works/{work_id}/synthesize")
async def synthesize_work_critiques(work_id: int):
    \"\"\"
    Synthesize all critiques for a work into a summary.
    Backward-compatible endpoint (returns summary string).
    \"\"\"
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")

    report = synthesizer.synthesize_work_critiques(work_id)

    return {
        "work_id": work_id,
        "summary": report.get("digest", "No critiques to synthesize."),
    }"""
new_synth_crit = """@app.get("/api/works/{work_id}/synthesize")
async def synthesize_work_critiques(work_id: int):
    work = collector.get_work(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="Work not found")
    critiques_data = db.get_all_critique_details_for_work(work_id)
    clusters = process_batch_2b_3(critiques_data)
    report = synthesizer.synthesize_clusters(clusters)
    return {"work_id": work_id, "summary": report.get("digest", "No critiques to synthesize.")}"""
if old_synth_crit in text:
    text = text.replace(old_synth_crit, new_synth_crit)
else:
    print("FAILED TO MATCH SYNTHESIZE")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(text)
print("done patching main")
