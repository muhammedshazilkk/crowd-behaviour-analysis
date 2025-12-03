# app.py
import os, time, io, json
from flask import Flask, Response, render_template, send_from_directory, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Paths
STREAM_PATH = os.path.join(os.getcwd(), "stream", "stream.jpg")
ALERTS_DIR = os.path.join(os.getcwd(), "alerts")
LOG_FILE = os.path.join(ALERTS_DIR, "log.jsonl")

# ---- helper to read log lines into JSON list ----
def read_logs():
    logs = []
    if not os.path.exists(LOG_FILE):
        return logs
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except Exception:
                # skip bad line
                continue
    # sort by time descending if 'time' exists in YYYYMMDD_HHMMSS format
    try:
        logs.sort(key=lambda x: x.get("time",""), reverse=True)
    except Exception:
        pass
    return logs

# ---- routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/snapshot")
def snapshot():
    # Return latest stream image
    if not os.path.exists(STREAM_PATH):
        abort(404)
    return send_from_directory(os.path.dirname(STREAM_PATH), os.path.basename(STREAM_PATH))

@app.route("/api/alerts")
def api_alerts():
    logs = read_logs()
    # augment with thumbnail path if image exists
    for entry in logs:
        img = entry.get("image")
        if img and os.path.exists(os.path.join(os.getcwd(), img)):
            entry["image_url"] = "/" + img.replace("\\", "/")
        else:
            entry["image_url"] = None
        # also add video_url if present and file exists
        if "video" in entry and os.path.exists(os.path.join(os.getcwd(), entry["video"])):
            entry["video_url"] = "/" + entry["video"].replace("\\", "/")
        else:
            entry["video_url"] = None
    return jsonify({"alerts": logs})

@app.route("/alerts/<path:filename>")
def serve_alerts_file(filename):
    # Serve files under alerts dir
    return send_from_directory(ALERTS_DIR, filename)

# Optional: Serve entire alerts directory listing (simple)
@app.route("/api/alerts/list")
def alerts_list():
    files = []
    if os.path.exists(ALERTS_DIR):
        for fname in sorted(os.listdir(ALERTS_DIR), reverse=True):
            files.append(fname)
    return jsonify(files)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
