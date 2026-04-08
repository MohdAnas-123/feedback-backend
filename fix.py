import re

with open("main.py", "r", encoding="utf-8") as f:
    text = f.read()

helpers = """
def process_critique_1_and_2a(critique_id: int, raw_text: str, aspect: str = "") -> dict:
    preprocessed = preprocessor.preprocess(raw_text)
    db.update_critique_preprocessing(
        critique_id, cleaned_text=preprocessed["cleaned_text"],
        is_meaningful=preprocessed["is_meaningful"]
    )
    if not preprocessed["is_meaningful"]: return {"critique_id": critique_id, "is_meaningful": False}

    analysis_2a = analyzer.analyze_critique_2a(preprocessed["cleaned_text"], preprocessed["embedding"], preprocessed.get("length", 0))
    db.update_critique_analysis(critique_id, tone_score=analysis_2a["quality_scores"]["tone"], actionability_score=analysis_2a["quality_scores"]["specificity"], sentiment_polarity=analysis_2a["sentiment"]["polarity"], sentiment_intensity=analysis_2a["sentiment"]["intensity"], intent=analysis_2a["intent"]["primary_intent"], quality_clarity=analysis_2a["quality_scores"]["clarity"], quality_specificity=analysis_2a["quality_scores"]["specificity"], quality_overall=analysis_2a["quality_scores"]["overall"])
    return {"critique_id": critique_id, "is_meaningful": True, "analysis": analysis_2a}

def process_batch_2b_3(critiques_data):
    if not critiques_data: return []
    embeddings_map = {c["id"]: preprocessor.vectorize(c.get("cleaned_text") or c.get("raw_text", "")) for c in critiques_data if c.get("is_meaningful")}
    embeddings_map = {k: v for k, v in embeddings_map.items() if v is not None and len(v) > 0}
            
    clustering = analyzer.batch_cluster_2b(embeddings_map)
    for cid in embeddings_map.keys(): db.update_critique_cluster(cid, clustering.get(cid, 0))
        
    clusters = {}
    for c in critiques_data:
        if not c.get("is_meaningful"): continue
        cid = c["id"]
        c_id = clustering.get(cid, 0)
        if c_id not in clusters: clusters[c_id] = {"cluster_id": c_id, "critiques": [], "intent_counts": {}, "sent_counts": {}, "quality_sum": 0.0, "total": 0}
        clusters[c_id]["critiques"].append(c["raw_text"])
        clusters[c_id]["quality_sum"] += c.get("quality_overall", 0.5)
        clusters[c_id]["total"] += 1
        clusters[c_id]["intent_counts"][c.get("intent", "observation")] = clusters[c_id]["intent_counts"].get(c.get("intent", "observation"), 0) + 1
        clusters[c_id]["sent_counts"][c.get("sentiment_polarity", "neutral")] = clusters[c_id]["sent_counts"].get(c.get("sentiment_polarity", "neutral"), 0) + 1

    final_clusters = []
    for c_id, c_data in clusters.items():
        total = max(1, c_data["total"])
        cluster_payload = {
            "cluster_id": c_id,
            "critiques": c_data["critiques"],
            "avg_sentiment": max(c_data["sent_counts"], key=c_data["sent_counts"].get),
            "dominant_intent": max(c_data["intent_counts"], key=c_data["intent_counts"].get),
            "avg_quality": c_data["quality_sum"] / total
        }
        cluster_payload.update(enhancer.process_cluster(cluster_payload))
        final_clusters.append(cluster_payload)

    return final_clusters
"""

old_pattern = r'def process_critique_pipeline\(critique_id: int, raw_text: str, aspect: str = ""\) -> dict:.*?@app\.post\("/api/agent/process"\)'

new_code = helpers + '''
@app.get("/api/works/{work_id}")
async def get_work(work_id: int):
    work = collector.get_work(work_id)
    if not work: raise HTTPException(status_code=404, detail="Work not found")
    return {"work": {"id": work[0], "user_id": work[1], "content": work[2], "type": work[3], "created_at": work[4]}, "critiques": db.get_all_critique_details_for_work(work_id)}

@app.get("/api/works/{work_id}/synthesize")
async def synthesize_work_critiques(work_id: int):
    work = collector.get_work(work_id)
    if not work: raise HTTPException(status_code=404, detail="Work not found")
    report = synthesizer.synthesize_clusters(process_batch_2b_3(db.get_all_critique_details_for_work(work_id)))
    return {"work_id": work_id, "summary": report.get("digest", "No critiques.")}

@app.get("/api/works/{work_id}/report")
async def generate_report(work_id: int):
    work = collector.get_work(work_id)
    if not work: raise HTTPException(status_code=404, detail="Work not found")
    report = synthesizer.synthesize_clusters(process_batch_2b_3(db.get_all_critique_details_for_work(work_id)))
    return {"work_id": work_id, "work_content": work[2], "work_type": work[3], "report": report}

@app.post("/api/process-feedback")
async def process_feedback(data: WorkWithCritiques):
    work_id = collector.add_work(data.work.user_id, data.work.content, data.work.type)
    if not work_id: raise HTTPException(status_code=500, detail="Failed to create work")
    for critique in data.critiques:
        cid = collector.add_critique(work_id, critique.aspect, critique.raw_text)
        if cid: process_critique_1_and_2a(cid, critique.raw_text, critique.aspect)
    report = synthesizer.synthesize_clusters(process_batch_2b_3(db.get_all_critique_details_for_work(work_id)))
    return {"work_id": work_id, "report": report}

@app.post("/api/agent/process")'''

text = re.sub(old_pattern, new_code, text, flags=re.DOTALL)

add_critique_pattern = r'@app\.post\("/api/works/\{work_id\}/critiques"\).*?def add_critique.*?return result'
new_add_critique = '''@app.post("/api/works/{work_id}/critiques")
async def add_critique(work_id: int, critique: Critique):
    critique_id = collector.add_critique(work_id, critique.aspect, critique.raw_text)
    if not critique_id: raise HTTPException(status_code=500, detail="Failed to add critique")
    return process_critique_1_and_2a(critique_id, critique.raw_text, critique.aspect)'''
text = re.sub(add_critique_pattern, new_add_critique, text, flags=re.DOTALL)

text = text.replace("from analyzer import BERTAnalyzer", "from analyzer import Stage2Analyzer")
text = text.replace("analyzer = BERTAnalyzer(db=db)", "analyzer = Stage2Analyzer(db=db)")
text = text.replace('print("✅ Gemini agent initialized")', 'print("SUCCESS: Gemini agent initialized")')

with open("main.py", "w", encoding="utf-8") as f:
    f.write(text)
print("done patching main")
