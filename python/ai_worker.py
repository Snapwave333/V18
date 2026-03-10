"""
Standalone AI worker script.
Launched as a subprocess by main.py.
Writes generated images as PNG files to a watch directory.
Reads audio intensity from a shared text file written by main.py.

Runs StreamDiffusion in a continuous loop with NO sleep — generates as fast
as the GPU allows. Each new frame is written atomically so the renderer always
has the freshest image available. The shader handles smooth visual crossfading.

Heavy imports are at module level so CUDA initializes during import,
not deferred into a function (which causes a hang on Windows).
"""
import sys
import os
import time
import json
import traceback

from ai_generator import AIGenerator
from susa import Susa
from storyteller import get_storyteller

# Re-evaluate audio and potentially pick a new prompt every N generated frames.
PROMPT_INTERVAL = 8


def read_features(features_file):
    """Read JSON audio features written by main.py audio_thread. Returns dict."""
    try:
        with open(features_file, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def run(watch_dir, features_file):
    os.makedirs(watch_dir, exist_ok=True)

    # FX state file — same directory as features_file
    fx_state_file = os.path.join(os.path.dirname(features_file), "vj_fx_state.json")
    # Initialize with empty state
    with open(fx_state_file, "w") as f:
        f.write("{}")

    try:
        print("[AI WORKER] Initializing AIGenerator...")
        ai_generator = AIGenerator()
        susa = Susa()
        storyteller = get_storyteller()
        storyteller.start()
        print("[AI WORKER] Ready.")
    except Exception as e:
        print(f"[AI WORKER] Init failed: {e}")
        traceback.print_exc()
        open(os.path.join(watch_dir, "FAIL"), "w").close()
        return

    frame_idx = 0
    frames_since_prompt = PROMPT_INTERVAL  # force prompt pick on first iteration
    current_prompt = None

    while True:
        if frames_since_prompt >= PROMPT_INTERVAL:
            features = read_features(features_file)
            narrative = storyteller.get_context()
            beat_name = storyteller.get_beat_name()
            new_prompt = susa.generate_prompt(features, narrative)
            if new_prompt != current_prompt:
                rms   = features.get("smoothed_rms", 0)
                beat  = features.get("beat", False)
                cent  = features.get("centroid", 0)
                print(
                    f"[AI] rms={rms:.0f}  beat={'Y' if beat else 'N'}  "
                    f"cent={cent:.2f}  story='{beat_name}'"
                )
                print(f"     prompt='{new_prompt[:80]}...'")
                current_prompt = new_prompt

            # Generate and write FX directives
            fx_state = susa.generate_fx_state(features, beat_name)
            try:
                tmp = fx_state_file + ".tmp"
                with open(tmp, "w") as f:
                    f.write(json.dumps(fx_state))
                os.replace(tmp, fx_state_file)
            except Exception:
                pass

            frames_since_prompt = 0

        try:
            image = ai_generator.generate_image(current_prompt)

            tmp_path = os.path.join(watch_dir, f"frame_{frame_idx}.tmp")
            out_path = os.path.join(watch_dir, f"frame_{frame_idx}.png")
            image.save(tmp_path, format="PNG")
            os.replace(tmp_path, out_path)

            if frame_idx > 0:
                prev = os.path.join(watch_dir, f"frame_{frame_idx - 1}.png")
                try:
                    os.remove(prev)
                except Exception:
                    pass

            frame_idx += 1
            frames_since_prompt += 1

        except Exception as e:
            print(f"[AI WORKER] Generation failed: {e}")
            traceback.print_exc()
            time.sleep(1.0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ai_worker.py <watch_dir> <features_file>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
