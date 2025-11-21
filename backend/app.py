# backend/app.py
import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request, g
import sqlite3
from model import train_model_from_dataset, predict_priority, load_model

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DB_PATH = BASE_DIR / "storage.db"
SAMPLE_PATH = BASE_DIR / "sample_dataset.json"
MODEL_PATH = BASE_DIR / "priority_model.json"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), template_folder=str(FRONTEND_DIR), static_url_path="")

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(str(DB_PATH))
    return db

def init_db():
    db = get_db()
    db.execute("""
    CREATE TABLE IF NOT EXISTS settings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      domain TEXT UNIQUE,
      priority TEXT,
      sound TEXT,
      vibration TEXT,
      volume INTEGER
    );
    """)
    db.commit()

@app.before_first_request
def setup():
    init_db()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/domain/<path:name>")
def domain_page(name):
    return send_from_directory(str(FRONTEND_DIR), "page_domain.html")

@app.route("/api/train", methods=["POST"])
def api_train():
    if not SAMPLE_PATH.exists():
        return jsonify({"error":"sample_dataset.json not found on server."}), 400
    try:
        data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"failed to read dataset: {str(e)}"}), 500
    try:
        res = train_model_from_dataset(data)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"training failed: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}
    if not payload:
        if not SAMPLE_PATH.exists():
            return jsonify({"error":"no dataset present on server"}), 400
        payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    try:
        order = predict_priority(payload, blend_alpha=0.6)
        return jsonify({"order": order})
    except Exception as e:
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500

@app.route("/api/settings/<domain>", methods=["POST"])
def save_settings(domain):
    payload = request.get_json() or {}
    priority = payload.get("priority","medium")
    sound = payload.get("sound","chime")
    vibration = payload.get("vibration","short")
    volume = int(payload.get("volume",70))
    db = get_db()
    db.execute("""
      INSERT INTO settings(domain,priority,sound,vibration,volume)
      VALUES(?,?,?,?,?)
      ON CONFLICT(domain) DO UPDATE SET
        priority=excluded.priority,
        sound=excluded.sound,
        vibration=excluded.vibration,
        volume=excluded.volume
    """, (domain, priority, sound, vibration, volume))
    db.commit()
    return jsonify({"status":"ok"})

@app.route("/api/settings/<domain>", methods=["GET"])
def load_settings(domain):
    db = get_db()
    cur = db.execute("SELECT priority,sound,vibration,volume FROM settings WHERE domain=?", (domain,))
    row = cur.fetchone()
    if row:
        return jsonify({"priority":row[0],"sound":row[1],"vibration":row[2],"volume":row[3]})
    return jsonify({})

@app.route("/api/reset", methods=["POST"])
def reset_all():
    db = get_db()
    db.execute("DELETE FROM settings")
    db.commit()
    model_file = MODEL_PATH
    if model_file.exists():
        try:
            model_file.unlink()
        except Exception:
            pass
    return jsonify({"status":"reset_ok","note":"settings cleared and model removed (sample preserved)."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
