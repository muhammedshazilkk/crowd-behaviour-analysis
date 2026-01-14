from flask import Flask, render_template, send_file, Response
import os
import time

app = Flask(__name__)

STREAM_PATH = os.path.join("stream", "stream.jpg")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    # 🔴 Wait until first frame exists
    if not os.path.exists(STREAM_PATH):
        return "Stream not ready", 404

    # 🔴 Disable caching fully
    return send_file(
        STREAM_PATH,
        mimetype="image/jpeg",
        cache_timeout=0,
        conditional=False
    )


if __name__ == "__main__":
    print("🚀 UI Server started at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

