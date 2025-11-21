# backend/model.py
from pathlib import Path
import json
import math
import random

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "priority_model.json"
SAMPLE_PATH = BASE_DIR / "sample_dataset.json"

# Features:
#  - bias (1)
#  - action_clicked (0/1)
#  - opened (0/1)
#  - dismissed (0/1)
#  - inv_delay = 1 / (1 + delay_seconds)  (higher => more immediate)
# We train a simple logistic regression via SGD across all event rows (pure python)

def load_dataset(path=SAMPLE_PATH):
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text)
    except Exception:
        return {}

def extract_features_and_labels(dataset):
    events = dataset.get("events", [])
    X = []
    y = []
    for e in events:
        action = int(e.get("action_clicked", 0))
        opened = int(e.get("opened", 0))
        dismissed = int(e.get("dismissed", 0))
        delay = float(e.get("delay_seconds", 0) or 0.0)
        inv_delay = 1.0 / (1.0 + delay)
        # label: did user meaningfully interact? (opened OR action_clicked)
        label = 1 if (opened or action) else 0
        features = [1.0, action, opened, dismissed, inv_delay]
        X.append(features)
        y.append(label)
    return X, y

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0

def init_weights(n):
    random.seed(42)
    return [random.uniform(-0.1, 0.1) for _ in range(n)]

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def train_sgd(X, y, lr=0.1, epochs=200):
    if not X:
        return None
    n_features = len(X[0])
    w = init_weights(n_features)
    for epoch in range(epochs):
        # shuffle
        combined = list(zip(X, y))
        random.shuffle(combined)
        for xi, yi in combined:
            pred = sigmoid(dot(w, xi))
            error = yi - pred
            # gradient step
            for j in range(n_features):
                w[j] += lr * error * xi[j]
        # optionally decay lr slightly
        lr *= 0.995
    return w

def train_model_from_dataset(dataset):
    X, y = extract_features_and_labels(dataset)
    w = train_sgd(X, y, lr=0.15, epochs=300)
    if w is None:
        return {"error":"no data"}
    model = {"weights": w}
    MODEL_PATH.write_text(json.dumps(model), encoding="utf-8")
    return {"status":"trained","n_samples": len(X)}

def load_model():
    if MODEL_PATH.exists():
        try:
            data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            return data.get("weights", [])
        except Exception:
            return []
    return []

def score_event_with_model(weights, event):
    if not weights:
        return None
    action = int(event.get("action_clicked", 0))
    opened = int(event.get("opened", 0))
    dismissed = int(event.get("dismissed", 0))
    delay = float(event.get("delay_seconds", 0) or 0.0)
    inv_delay = 1.0 / (1.0 + delay)
    features = [1.0, action, opened, dismissed, inv_delay]
    val = dot(weights, features)
    return sigmoid(val)

def heuristic_scores_from_dataset(dataset):
    """
    Compute a deterministic per-domain heuristic score from events.
    Use: avg(interaction rate) * factor + recentness factor
    """
    events = dataset.get("events", [])
    domains = {}
    for e in events:
        d = e.get("domain")
        if d not in domains:
            domains[d] = {"count":0,"acted":0,"recent_min_delay": None}
        domains[d]["count"] += 1
        acted = 1 if (e.get("opened",0) or e.get("action_clicked",0)) else 0
        domains[d]["acted"] += acted
        delay = float(e.get("delay_seconds",0) or 0.0)
        if domains[d]["recent_min_delay"] is None or delay < domains[d]["recent_min_delay"]:
            domains[d]["recent_min_delay"] = delay
    # compute score
    scores = {}
    for d, v in domains.items():
        rate = v["acted"] / max(1, v["count"])
        recent_score = 1.0 / (1.0 + (v["recent_min_delay"] or 0.0))
        score = rate * 1.2 + recent_score * 0.8 + math.log1p(v["count"]) * 0.03
        scores[d] = float(score)
    return scores

def predict_priority(dataset, blend_alpha=0.6):
    """
    Hybrid prediction combining:
      heuristic score (always), and
      model-based per-event predictions averaged per-domain (if model exists)
    blend_alpha is weight given to model (0..1)
    """
    aggs = heuristic_scores_from_dataset(dataset)
    model_weights = load_model()
    model_preds_per_domain = {}
    if model_weights:
        # score every event with model and average per domain
        events = dataset.get("events", [])
        per_domain = {}
        for e in events:
            d = e.get("domain")
            s = score_event_with_model(model_weights, e)
            if s is None:
                continue
            if d not in per_domain:
                per_domain[d] = {"sum":0.0,"n":0}
            per_domain[d]["sum"] += s
            per_domain[d]["n"] += 1
        for d, v in per_domain.items():
            model_preds_per_domain[d] = v["sum"] / max(1, v["n"])
    # compose final scores
    final = {}
    domains = list(aggs.keys())
    for d in domains:
        h = aggs.get(d, 0.0)
        m = model_preds_per_domain.get(d, None)
        if m is not None:
            final_score = blend_alpha * m + (1.0 - blend_alpha) * h
        else:
            final_score = h
        final[d] = float(final_score)
    # sort descending
    ordered = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return [d for d,_ in ordered]
