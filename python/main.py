import threading
import time
import traceback
import subprocess
import sys
import os

# Early SDL initialization for window placement
os.environ['SDL_VIDEO_CENTERED'] = '1'

from prompt_generator import generate_prompt
import tempfile
import json
import logging
from agent_bridge import fetch_agent_state, get_fallback_state
from collections import deque
from PIL import Image
import numpy as np
from shared_memory_ipc import SharedFrameMemory

try:
    from loading_screen import CinematicLoadingScreen

    LOADING_SCREEN_AVAILABLE = True
except ImportError:
    LOADING_SCREEN_AVAILABLE = False
    print("Loading screen not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("vj_engine.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Configuration
FRAME_DELAY_SECONDS = 0  # No delay by default
MAX_FRAME_BUFFER_SIZE = 500  # Store more frames for smooth playback

_latest_features = {}
_features_lock = threading.Lock()

# Frame buffer for delayed playback
_frame_buffer = deque()
_frame_buffer_lock = threading.Lock()
_buffer_ready = threading.Event()
_buffer_startup = threading.Event()

# Error tracking
_error_count = 0
_last_error_time = 0


def audio_thread(audio_ingest, features_file, stop_event):
    global _latest_features
    last_prompt_time = 0
    
    # Initial state
    current_state = get_fallback_state()
    current_state["video_prompt"] = current_state.get("video_prompt", "a cinematic abstract background")
    current_state["motion_intensity"] = 0.5
    
    print(f"[AudioThread] AI Agent initialized.")

    while not stop_event.is_set():
        feat = audio_ingest.get_features()
        d = feat.to_dict()

        is_high_energy_beat = (
            d.get("beat", False)
            and d.get("beat_strength", 0.0) > 0.7
            and d.get("smoothed_rms", 0.0) > 3000
        )

        if time.time() - last_prompt_time > 10.0 or is_high_energy_beat:
            # Call the AI Agent
            logger.info("[AudioThread] Fetching new state from AI Agent...")
            new_state = fetch_agent_state(d)
            if new_state:
                current_state = new_state
            last_prompt_time = time.time()
            logger.info(f"[AudioThread] Agent Macro: {current_state.get('macro_state')}")

        # Inject agent state into features
        d.update(current_state)
        # SVD worker specifically looks for 'prompt' key
        d["prompt"] = current_state.get("video_prompt", "cinematic abstract")

        with _features_lock:
            _latest_features = d
        try:
            blob = json.dumps(d)
            tmp = features_file + ".tmp"
            with open(tmp, "w") as f:
                f.write(blob)
            os.replace(tmp, features_file)
        except Exception:
            pass


def get_latest_features():
    with _features_lock:
        return dict(_latest_features)


def add_frame_to_buffer(image, frame_id):
    """Add a new frame to the delay buffer with timestamp."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Brightness Diagnostic
    try:
        avg_v = np.array(image).mean()
        if avg_v > 250: # Pure white
            logger.warning(f"Frame {frame_id} is a blank white placeholder!")
        elif avg_v < 1:
            logger.warning(f"Frame {frame_id} is a blank black placeholder!")
    except:
        pass

    with _frame_buffer_lock:
        # Add frame with generation timestamp
        _frame_buffer.append(
            {"image": image, "frame_id": frame_id, "timestamp": time.time()}
        )

        # Remove old frames beyond buffer size
        while len(_frame_buffer) > MAX_FRAME_BUFFER_SIZE:
            oldest = _frame_buffer.popleft()
            try:
                if oldest["image"] is not None and hasattr(oldest["image"], "close"):
                    oldest["image"].close()
            except:
                pass

        # Check if we have enough aged frames for playback
        if not _buffer_ready.is_set() and len(_frame_buffer) >= 10:
            # Check if oldest frame is old enough
            current_time = time.time()
            oldest_age = (
                current_time - _frame_buffer[0]["timestamp"] if _frame_buffer else 0
            )
            if oldest_age >= FRAME_DELAY_SECONDS:
                _buffer_ready.set()
                print(
                    f"[FrameBuffer] Buffer ready! Oldest frame age: {oldest_age:.1f}s"
                )


def get_delayed_frame():
    """Get the oldest frame that's older than FRAME_DELAY_SECONDS."""
    current_time = time.time()

    with _frame_buffer_lock:
        if not _frame_buffer:
            return None

        # Find the first frame that's old enough
        for i, frame_data in enumerate(_frame_buffer):
            age = current_time - frame_data["timestamp"]
            if age >= FRAME_DELAY_SECONDS:
                # Return this frame (and remove older ones)
                result = frame_data["image"].copy()  # Copy to prevent issues
                # Remove all frames up to and including this one
                for _ in range(i + 1):
                    try:
                        old = _frame_buffer.popleft()
                        if old["image"] is not None and hasattr(old["image"], "close"):
                            old["image"].close()
                    except:
                        pass
                return result

        # No frame is old enough yet
        return None

def poll_shm_or_disk(last_seen_id, watch_dir, shm_reader_state):
    """Returns (image, last_seen_id, frame_id)"""
    shm_reader = shm_reader_state.get("shm_reader")
    if shm_reader is None:
        try:
            shm_reader = SharedFrameMemory("vj_frame_buffer", 448, 256, create=False)
            shm_reader_state["shm_reader"] = shm_reader
            logger.info("Connected to shared memory frame buffer for high-speed IPC")
        except Exception:
            pass
            
    if shm_reader:
        try:
            frame_np, frame_id, ts = shm_reader.read_frame()
            if frame_np is not None:
                if frame_id != last_seen_id:
                    # Return numpy array directly
                    return frame_np, frame_id, frame_id
                else:
                    return None, last_seen_id, None
        except Exception as e:
            logger.error(f"SHM read error: {e}")
            
    # Fallback to disk
    return watch_for_image(watch_dir, last_seen_id)


def get_buffer_status():
    """Get current buffer status for monitoring."""
    with _frame_buffer_lock:
        if not _frame_buffer:
            return "empty", 0, 0

        current_time = time.time()
        oldest_age = current_time - _frame_buffer[0]["timestamp"]
        newest_age = current_time - _frame_buffer[-1]["timestamp"]

        if oldest_age < FRAME_DELAY_SECONDS:
            return "buffering", len(_frame_buffer), oldest_age
        else:
            return "playing", len(_frame_buffer), oldest_age


def watch_for_image(watch_dir, last_seen):
    """Returns (PIL.Image, filename, frame_id) if a new PNG appeared, else (None, last_seen, None)."""
    try:
        files = [f for f in os.listdir(watch_dir) if f.endswith(".png") and f.startswith("frame_")]
    except Exception:
        return None, last_seen, None
    
    if not files:
        return None, last_seen, None

    # Numerical sort to handle non-padded or differently padded filenames
    import re
    def get_id(f):
        match = re.search(r'frame_(\d+)', f)
        return int(match.group(1)) if match else -1

    files.sort(key=get_id)
    latest = files[-1]
    
    if latest == last_seen:
        return None, last_seen, None

    frame_id = get_id(latest)
    path = os.path.join(watch_dir, latest)
    
    try:
        # Load and convert to RGB
        img = Image.open(path).convert("RGB").copy()
        return img, latest, frame_id
    except Exception:
        return None, last_seen, None


def read_fx_state(fx_state_file):
    """Read JSON FX directives written by ai_worker.py. Returns dict."""
    try:
        with open(fx_state_file, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}

def cleanup_old_frames(watch_dir, last_id, safety_margin=50):
    """
    Deletes files in watch_dir with frame IDs significantly older than last_id.
    Safety margin ensures we don't delete something the renderer might still need.
    """
    if not os.path.exists(watch_dir):
        return
    
    try:
        files = os.listdir(watch_dir)
        for f in files:
            if f.endswith(".png") and f.startswith("frame_"):
                try:
                    # Extract ID from frame_0000.png or frame_lastseen.png
                    # If it's a full filename like 'frame_000123.png'
                    fid_str = f.replace("frame_", "").replace(".png", "")
                    if "_" in fid_str: # handle possible double underscores or other naming
                         fid_str = fid_str.split("_")[-1]
                    fid = int(fid_str)
                    if fid < (last_id - safety_margin):
                        file_path = os.path.join(watch_dir, f)
                        os.remove(file_path)
                except (ValueError, OSError):
                    pass
    except Exception as e:
        logger.warning(f"Frame cleanup error: {e}")



def main(worker_type="optimized"):
    import pygame
    from audio_ingest import AudioIngest
    from shader_renderer import ShaderRenderer

    global _error_count

    audio_ingest = None
    renderer = None
    ai_proc = None
    stop_event = threading.Event()

    # ── FRESH START: Nuke all old frames and state ──
    ipc_dir = os.path.join(os.path.dirname(__file__), "temp_ipc")
    watch_dir = os.path.join(ipc_dir, "frames")
    features_file = os.path.join(ipc_dir, "audio_features.json")
    fx_state_file = os.path.join(ipc_dir, "vj_fx_state.json")

    import shutil
    if os.path.exists(ipc_dir):
        try:
            shutil.rmtree(ipc_dir)
            logger.info("Purged all old frames and state — fresh start")
        except Exception as e:
            logger.warning(f"Could not purge temp_ipc: {e}")

    os.makedirs(watch_dir, exist_ok=True)
    with open(features_file, "w") as f:
        f.write("{}")

    try:
        logger.info("=" * 60)
        logger.info("VIBES V18 — Fast Boot")
        logger.info(f"Worker: {worker_type} | Delay: {FRAME_DELAY_SECONDS}s")
        logger.info("=" * 60)

        worker_script_name = f"ai_worker_{worker_type}.py"
        worker_script = os.path.join(os.path.dirname(__file__), worker_script_name)
        logger.info(f"Launching AI worker: {worker_script}")

        python_executable = os.path.join(
            os.path.dirname(__file__), ".venv", "Scripts", "python.exe"
        )

        if not os.path.exists(python_executable):
            logger.warning(f"Venv python not found, trying system python")
            python_executable = "python"

        # Write worker output to a log file — avoids pipe buffer blocking
        worker_log_path = os.path.join(ipc_dir, "worker.log")
        worker_log_file = open(worker_log_path, "wb")
        ai_proc = subprocess.Popen(
            [python_executable, "-u", worker_script, watch_dir, features_file],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=worker_log_file,
            stderr=worker_log_file,
        )

        # Thread to tail worker log and forward to main logger
        def log_ai_output():
            try:
                with open(worker_log_path, "r", errors="replace") as f:
                    while ai_proc.poll() is None:
                        line = f.readline()
                        if line:
                            logger.info(f"[AI WORKER] {line.rstrip()}")
                        else:
                            import time as _t; _t.sleep(0.05)
            except Exception:
                pass

        threading.Thread(target=log_ai_output, daemon=True).start()

        # ── FAST BOOT: Initialize renderer + audio immediately ──
        logger.info("Initializing ShaderRenderer...")
        try:
            renderer = ShaderRenderer()
            logger.info("ShaderRenderer OK")
        except Exception as e:
            logger.error(f"ShaderRenderer failed: {e}")
            return

        logger.info("Initializing AudioIngest...")
        try:
            audio_ingest = AudioIngest()
        except Exception as e:
            logger.warning(f"AudioIngest failed: {e}. Running without audio.")
            audio_ingest = None

        if audio_ingest:
            threading.Thread(
                target=audio_thread,
                args=(audio_ingest, features_file, stop_event),
                daemon=True,
            ).start()

        # ── Wait for first frame from AI worker (fast poll) ──
        last_seen = None
        shm_reader_state = {"shm_reader": None}
        boot_start = time.time()
        boot_deadline = boot_start + 300  # 5 min timeout for first frame
        first_frame = None

        logger.info("Waiting for first AI frame...")
        
        while time.time() < boot_deadline:
            if ai_proc.poll() is not None:
                logger.error(f"AI worker exited with code {ai_proc.returncode}")
                return
            
            # Keep pumping Pygame events and update HUD status
            try:
                renderer.render(None, status="WARMING UP AI WORKERS...")
            except Exception:
                pass
                
            try:
                img, last_seen, frame_id = poll_shm_or_disk(last_seen, watch_dir, shm_reader_state)
                if img is not None:
                    add_frame_to_buffer(img, frame_id)
                    first_frame = img
                    logger.info(f"First frame received in {time.time() - boot_start:.1f}s")
                    break
            except Exception:
                pass
            # Pump pygame events handled by renderer.render above
            time.sleep(0.05)

        if first_frame is None:
            logger.error("No frames received from AI worker within timeout")
            return

        # Render first real frame immediately
        try:
            renderer.render(first_frame, get_latest_features())
        except Exception as e:
            logger.warning(f"First frame render error: {e}")

        logger.info("SHOWTIME")

        running = True
        clock = pygame.time.Clock()
        loop_count = 0
        last_status_log = time.time()
        fps_history = deque(maxlen=60)
        new_image = None
        fx_state = {}

        while running:
            loop_start = time.time()

            # ── Poll for new AI frames (every 3rd iteration to reduce I/O) ──
            if loop_count % 3 == 0:
                try:
                    new_image, last_seen, frame_id = poll_shm_or_disk(last_seen, watch_dir, shm_reader_state)
                    if new_image is not None:
                        add_frame_to_buffer(new_image, frame_id)
                except Exception as e:
                    logger.warning(f"Error watching for image: {e}")

            # ── Periodic disk cleanup ──
            if loop_count % 300 == 0 and last_seen and last_seen > 0:
                cleanup_old_frames(watch_dir, last_seen, safety_margin=150)

            # ── Get delayed frame for display ──
            delayed_frame = get_delayed_frame()

            # ── Audio (thread-safe, fast) ──
            try:
                audio = get_latest_features() if audio_ingest else {}
            except Exception:
                audio = {}

            # ── FX state (read from disk every 30th frame) ──
            if loop_count % 30 == 0:
                try:
                    fx_state = read_fx_state(fx_state_file)
                    renderer.apply_ai_fx(fx_state)
                except Exception:
                    pass

            # ── Render ──
            display_image = delayed_frame if delayed_frame is not None else new_image
            try:
                running = renderer.render(display_image, audio, status="OK (GENERATING)")
                if not running:
                    logger.info("renderer.render() returned False — user closed window or pressed ESC/Q")
            except Exception as e:
                logger.error(f"Render error: {e}")
                _error_count += 1
                if _error_count > 10:
                    logger.error("Too many render errors, stopping")
                    running = False
                else:
                    time.sleep(0.05)

            # ── FPS tracking ──
            clock.tick(60)
            loop_count += 1
            loop_time = time.time() - loop_start
            fps_history.append(loop_time)

            # Update window title with FPS every 30 frames
            if loop_count % 30 == 0:
                avg = sum(fps_history) / len(fps_history) if fps_history else 1
                fps = 1.0 / avg if avg > 0 else 0
                pygame.display.set_caption(f"V18  |  {fps:.0f} FPS")

            # Periodic status logging
            if time.time() - last_status_log > 10:
                status, count, age = get_buffer_status()
                avg = sum(fps_history) / len(fps_history) if fps_history else 1
                fps = 1.0 / avg if avg > 0 else 0
                logger.info(f"Play: {status} | Buf: {count} | Delay: {age:.1f}s | FPS: {fps:.1f}")
                last_status_log = time.time()

        logger.info(f"Main run loop finished after {loop_count} iterations.")

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"FATAL: {e}")
        traceback.print_exc()
    finally:
        logger.info("Cleaning up...")
        stop_event.set()

        if ai_proc is not None and ai_proc.poll() is None:
            ai_proc.terminate()
            try:
                ai_proc.wait(timeout=3)
            except Exception:
                ai_proc.kill()

        if audio_ingest is not None:
            try:
                audio_ingest.stop()
            except Exception as e:
                logger.warning(f"Error stopping audio: {e}")

        if renderer is not None:
            try:
                renderer.stop()
            except Exception as e:
                logger.warning(f"Error stopping renderer: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker", type=str, default="deform", help="Worker type: deform (default), optimized, video"
    )
    parser.add_argument("--delay", type=int, default=0, help="Frame delay in seconds")
    args = parser.parse_args()
    FRAME_DELAY_SECONDS = args.delay
    main(worker_type=args.worker)
