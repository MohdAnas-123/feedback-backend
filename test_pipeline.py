"""
End-to-end test for the CritiqueConnect pipeline.
Tests all 4 stages of the paper architecture.
"""

import requests
import json
import time
import sys

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def test(name, response, expected_status=200):
    global PASS, FAIL
    status = response.status_code
    ok = status == expected_status

    if ok:
        PASS += 1
        print(f"  ✅ {name} — {status}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — expected {expected_status}, got {status}")
        try:
            print(f"     Body: {response.json()}")
        except Exception:
            print(f"     Body: {response.text[:300]}")

    return ok


def pp(data):
    """Pretty-print JSON"""
    print(json.dumps(data, indent=2, default=str))


# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CRITIQUECONNECT END-TO-END PIPELINE TEST")
print("=" * 60)

# ─── Test 0: Health check ─────────────────────
print("\n--- Health Check ---")
r = requests.get(f"{BASE}/")
if test("Root endpoint", r):
    data = r.json()
    assert "pipeline" in data, "Missing pipeline info"
    print(f"     Pipeline: {data['pipeline']}")

r = requests.get(f"{BASE}/api/hello")
test("Hello endpoint", r)

# ─── Test 1: Create a work ────────────────────
print("\n--- Stage 0: Create Work ---")
work_data = {
    "user_id": "test_user_123",
    "content": "A minimalist landing page with blue and white color scheme, featuring a hero section with bold typography and a call-to-action button.",
    "type": "web_design"
}
r = requests.post(f"{BASE}/api/works", json=work_data)
if test("Create work", r):
    work_id = r.json()["work_id"]
    print(f"     Work ID: {work_id}")
else:
    print("FATAL: Cannot continue without work_id")
    sys.exit(1)

# ─── Test 2: Add critiques (runs Stages 1-3) ──
print("\n--- Stages 1→2→3: Add & Process Critiques ---")

critiques = [
    {
        "aspect": "color scheme",
        "raw_text": "The blue and white colors clash and look unprofessional. Consider using a more harmonious palette with complementary tones."
    },
    {
        "aspect": "typography",
        "raw_text": "The font size is too small for the hero section. Increase it to at least 48px for better readability on mobile devices."
    },
    {
        "aspect": "layout",
        "raw_text": "The call-to-action button is buried below the fold. Move it above the fold and make it more prominent with a contrasting color."
    },
    {
        "aspect": "spacing",
        "raw_text": "Too much whitespace between sections makes the page feel disconnected. Tighten the spacing to create a more cohesive flow."
    },
    {
        "aspect": "overall",
        "raw_text": "Good job"  # This should be filtered as non-meaningful
    },
]

meaningful_count = 0
filtered_count = 0

for i, critique in enumerate(critiques):
    r = requests.post(f"{BASE}/api/works/{work_id}/critiques", json=critique)
    if test(f"Critique {i+1} ({critique['aspect']})", r):
        result = r.json()

        if result.get("is_meaningful"):
            meaningful_count += 1
            # Verify Stage 2 analysis fields
            analysis = result.get("analysis", {})
            if analysis:
                sentiment = analysis.get("sentiment", {})
                intent = analysis.get("intent", {})
                quality = analysis.get("quality", {})
                print(f"     Sentiment: {sentiment.get('polarity')} ({sentiment.get('intensity')})")
                print(f"     Intent: {intent.get('primary_intent')} ({intent.get('confidence')})")
                print(f"     Quality: {quality.get('overall')}")
            
            # Verify Stage 3 enhancement
            enhanced = result.get("enhanced_text")
            if enhanced:
                print(f"     Enhanced: {enhanced[:100]}...")
        else:
            filtered_count += 1
            print(f"     ⚡ Filtered: {result.get('message')}")

print(f"\n     Summary: {meaningful_count} meaningful, {filtered_count} filtered")

# ─── Test 3: Get work with full details ───────
print("\n--- Retrieve Work + Critiques ---")
r = requests.get(f"{BASE}/api/works/{work_id}")
if test("Get work details", r):
    data = r.json()
    print(f"     Work type: {data['work']['type']}")
    print(f"     Critiques count: {len(data['critiques'])}")
    
    # Verify expanded analysis fields in critique records
    if data['critiques']:
        c = data['critiques'][0]
        fields = ['sentiment_polarity', 'sentiment_intensity', 'intent',
                  'quality_clarity', 'quality_specificity', 'quality_overall']
        present = [f for f in fields if f in c]
        print(f"     Analysis fields present: {len(present)}/{len(fields)}")

# ─── Test 4: Backward-compatible synthesize ───
print("\n--- Stage 3: Synthesize (backward-compatible) ---")
r = requests.get(f"{BASE}/api/works/{work_id}/synthesize")
if test("Synthesize critiques", r):
    data = r.json()
    summary = data.get("summary", "")
    print(f"     Summary: {summary[:150]}...")

# ─── Test 5: Full structured report (Paper Stage 4) ──
print("\n--- Stage 4: Full Structured Report ---")
r = requests.get(f"{BASE}/api/works/{work_id}/report")
if test("Generate report", r):
    data = r.json()
    report = data.get("report", {})
    
    print(f"\n     📊 REPORT FOR WORK {work_id}")
    print(f"     {'─' * 40}")
    
    # Digest
    print(f"     Digest: {report.get('digest', 'N/A')[:150]}")
    
    # Sentiment overview
    sentiment = report.get("sentiment_overview", {})
    print(f"     Sentiment: +{sentiment.get('positive', 0)}% / ={sentiment.get('neutral', 0)}% / -{sentiment.get('negative', 0)}%")
    
    # Strengths
    strengths = report.get("strengths", [])
    print(f"     Strengths ({len(strengths)}):")
    for s in strengths[:3]:
        print(f"       • {s[:100]}")
    
    # Weaknesses
    weaknesses = report.get("weaknesses", [])
    print(f"     Weaknesses ({len(weaknesses)}):")
    for w in weaknesses[:3]:
        print(f"       • {w[:100]}")
    
    # Recommendations
    recommendations = report.get("recommendations", [])
    print(f"     Recommendations ({len(recommendations)}):")
    for rec in recommendations[:3]:
        if isinstance(rec, dict):
            print(f"       [{rec.get('impact', '?')}] {rec.get('text', '')[:100]}")
        else:
            print(f"       • {str(rec)[:100]}")
    
    # Clustering
    clustering = report.get("clustering", {})
    print(f"     Clusters: {clustering.get('n_clusters', '?')} (silhouette: {clustering.get('silhouette_score', '?')})")
    
    # Quality
    print(f"     Avg Quality: {report.get('average_quality', 'N/A')}")
    print(f"     Critique Count: {report.get('critique_count', 'N/A')}")

# ─── Test 6: Process-feedback (all-in-one) ────
print("\n--- All-in-One: process-feedback ---")
all_in_one = {
    "work": {
        "user_id": "demo_user",
        "content": "Mobile app for task management with dark mode and gesture navigation",
        "type": "app_design"
    },
    "critiques": [
        {"aspect": "UX", "raw_text": "The swipe gestures are not intuitive. Users expect right-swipe to complete a task, not delete it."},
        {"aspect": "theme", "raw_text": "Dark mode colors are too harsh. Use softer grays instead of pure black for better eye comfort."},
    ]
}
r = requests.post(f"{BASE}/api/process-feedback", json=all_in_one)
if test("Process feedback all-in-one", r):
    data = r.json()
    print(f"     Work ID: {data.get('work_id')}")
    print(f"     Results: {len(data.get('results', []))} critiques processed")
    report = data.get("report")
    if report:
        print(f"     Report digest: {report.get('digest', 'N/A')[:100]}...")

# ─── Test 7: Agent endpoint ───────────────────
print("\n--- Gemini Agent Direct ---")
agent_data = {
    "id": "test_001",
    "title": "Online Code Editor",
    "description": "A web-based code editor with syntax highlighting and auto-complete",
    "reviews": [
        "The syntax highlighting is great but auto-complete is slow and often irrelevant",
        "Love the dark theme but wish there were more language support options",
        "Tab management is confusing, hard to switch between multiple files quickly"
    ]
}
r = requests.post(f"{BASE}/api/agent/process", json=agent_data)
if test("Agent process", r):
    data = r.json()
    print(f"     Summary: {data.get('overall_summary', 'N/A')[:150]}...")
    sentiment = data.get("sentiment_analysis", {})
    print(f"     Sentiment: {sentiment}")

# ─── Test 8: Edge cases ──────────────────────
print("\n--- Edge Cases ---")

# Non-existent work
r = requests.get(f"{BASE}/api/works/99999")
test("Non-existent work (404)", r, expected_status=404)

# Non-existent report
r = requests.get(f"{BASE}/api/works/99999/report")
test("Non-existent report (404)", r, expected_status=404)

# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60 + "\n")

if FAIL > 0:
    sys.exit(1)
