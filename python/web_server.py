from flask import Flask, jsonify, render_template, send_file
from agent_bridge import fetch_agent_state, get_fallback_state
import time
import math
import random
import os
import io
import numpy as np
from PIL import Image
from shared_memory_ipc import SharedFrameMemory

app = Flask(__name__)

TEMP_IPC_DIR = os.path.join(os.path.dirname(__file__), "temp_ipc")
BUFFER_STATUS_FILE = os.path.join(TEMP_IPC_DIR, "buffer_status.json")
FRAMES_DIR = os.path.join(TEMP_IPC_DIR, "frames")

# Track which frame to serve next
_frame_index = 0

# Shared memory reader
_shm_reader = None


def generate_dynamic_audio_telemetry():
    current_time = time.time()
    bpm = 130.0 + (math.sin(current_time * 0.1) * 10.0)
    sub_bass_energy = (
        0.5 + (math.sin(current_time * 0.5) * 0.4) + (random.random() * 0.1)
    )
    sub_bass_energy = max(0, min(1, sub_bass_energy))
    transients = [random.random() for _ in range(4)]
    if random.random() > 0.8:
        transients[random.randint(0, 3)] = 0.95
    return {
        "bpm": round(bpm, 2),
        "sub_bass_energy": round(sub_bass_energy, 4),
        "transients": transients,
    }


def get_buffer_status():
    try:
        if os.path.exists(BUFFER_STATUS_FILE):
            with open(BUFFER_STATUS_FILE, "r") as f:
                import json

                data = json.load(f)
                return data.get("ready", False)
    except:
        pass
    return False


def get_delayed_frame():
    """Get the latest available frame from the worker."""
    try:
        if not os.path.exists(FRAMES_DIR):
            return None

        files = [f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")]
        if not files:
            return None

        # Sort by frame number and get the absolute latest
        files.sort(
            key=lambda x: (
                int(x.split("_")[1].split(".")[0]) if "_" in x and ".png" in x else 0
            )
        )
        
        return os.path.join(FRAMES_DIR, files[-1])

    except Exception as e:
        print(f"Error getting frame: {e}")

    return None

def get_shm_frame():
    global _shm_reader
    if _shm_reader is None:
        try:
            _shm_reader = SharedFrameMemory("vj_frame_buffer", 512, 288, create=False)
        except Exception:
            return None
    
    try:
        frame_np, frame_id, ts = _shm_reader.read_frame()
        if frame_np is not None:
            return frame_np
    except Exception:
        _shm_reader = None # Reset on error
        
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/vj_state")
def get_vj_state():
    # Read the real state from audio_features.json (managed by main.py)
    features_file = os.path.join(TEMP_IPC_DIR, "audio_features.json")
    state = {
        "status": "OFFLINE",
        "current_beat": "IDLE",
        "narrative": "Waiting for engine...",
        "bpm": 0,
        "energy": 0,
        "buffer_ready": get_buffer_status()
    }
    
    try:
        if os.path.exists(features_file):
            with open(features_file, "r") as f:
                import json
                data = json.load(f)
                state["status"] = "OK"
                # Map internal keys to frontend keys
                state["current_beat"] = data.get("current_beat") or data.get("macro_state") or "STORY PROGRESSING"
                state["narrative"] = data.get("video_prompt") or data.get("prompt") or "..."
                state["bpm"] = data.get("bpm", 128)
                state["energy"] = data.get("smoothed_rms", 0) / 10000.0 # Scale to 0-1 approx
                state["energy"] = max(0, min(1, state["energy"]))
    except Exception as e:
        print(f"Error reading features: {e}")

    return jsonify(state)


@app.route("/api/latest_frame")
@app.route("/api/frame")
def get_frame():
    # Try shared memory first
    frame_np = get_shm_frame()
    if frame_np is not None:
        try:
            img = Image.fromarray(frame_np)
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
        except Exception as e:
            print(f"shm to png error: {e}")

    # Fallback to disk
    frame_path = get_delayed_frame()
    if frame_path and os.path.exists(frame_path):
        return send_file(frame_path, mimetype="image/png")
    return "No frame", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
