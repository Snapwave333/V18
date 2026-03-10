"""
Storyteller — Dan Harmon Story Circle narrative engine.

Runs as a background thread. Every BEAT_INTERVAL seconds it advances one
step around the Dan Harmon Story Circle (8 beats) and uses Ollama to
generate a short visual description that embodies that beat's emotional energy.

The current narrative context is exposed via get_context() as a short string
that Susa injects into the SD prompt alongside audio-reactive vocabulary.

Dan Harmon Story Circle beats:
  1. YOU       — a character in a zone of comfort
  2. NEED       — they need something
  3. GO         — they enter an unfamiliar world
  4. SEARCH     — they adapt and search
  5. FIND       — they find what they needed
  6. TAKE       — but they pay a heavy price for it
  7. RETURN     — and return to their familiar world
  8. CHANGE     — having been changed forever

Each beat maps to a visual archetype that SD can render well.
"""

import threading
import time
import subprocess
import json
import os
from collections import deque

# Persistent storage for what worked
_DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
_MEMORY_FILE = os.path.join(_DATA_DIR, "storyteller_memory.json")

# ── Story circle definition ───────────────────────────────────────────────────

STORY_BEATS = [
    {
        "name": "YOU — The Old Library",
        "energy": "calm",
        "archetype": (
            "an elderly librarian reading by a green desk lamp in a vast, dust-filled library, "
            "towering wooden shelves, warm incandescent glow, comfortable solitude"
        ),
    },
    {
        "name": "NEED — The Glowing Map",
        "energy": "yearning",
        "archetype": (
            "a hooded traveler studying a glowing holographic map in a dark stone room, "
            "determination on their face, a single blue crystal lighting the scene"
        ),
    },
    {
        "name": "GO — The Forest Gate",
        "energy": "threshold",
        "archetype": (
            "a traveler stepping through a massive stone archway overgrown with glowing vines, "
            "transition from a sunny meadow into a dark magical forest, mystery"
        ),
    },
    {
        "name": "SEARCH — The Sunken City",
        "energy": "searching",
        "archetype": (
            "an explorer swimming through the ruins of a submerged marble palace, "
            "shafts of sunlight piercing through blue water, schools of fish, hidden treasure"
        ),
    },
    {
        "name": "FIND — The Ornate Key",
        "energy": "revelation",
        "archetype": (
            "a pair of weathered hands holding a brilliant golden key, "
            "radiant light reflecting off intricate engravings, a moment of profound discovery"
        ),
    },
    {
        "name": "TAKE — The Price of Power",
        "energy": "sacrifice",
        "archetype": (
            "a figure in a dark ritual chamber, their hands glowing with unstable energy, "
            "cracks appearing in the ground, a sense of heavy sacrifice and dark beauty"
        ),
    },
    {
        "name": "RETURN — The Mountain Pass",
        "energy": "return",
        "archetype": (
            "a weary traveler walking along a high mountain ridge at sunset, "
            "holding a staff, looking down at a distant cottage with a smoking chimney"
        ),
    },
    {
        "name": "CHANGE — The Transformed Healer",
        "energy": "transcendence",
        "archetype": (
            "a young healer standing in a vibrant garden, flowers blooming at their touch, "
            "a peaceful aura, the character changed and empowered by their journey"
        ),
    },
]

# Seconds between advancing to the next story beat
BEAT_INTERVAL = 30.0

# Ollama model to use for narrative generation (fast local model)
OLLAMA_MODEL = "llama3.2"

# How many words the generated visual description should be (approx)
TARGET_WORDS = 20


def _call_ollama(prompt: str) -> str:
    """Call Ollama via subprocess. Returns generated text or empty string on failure."""
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _ollama_api(prompt: str) -> str:
    """Try Ollama HTTP API (faster than subprocess). Falls back to empty."""
    try:
        import urllib.request
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.9, "num_predict": 50},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception:
        return ""


def _generate_visual(beat: dict, audio_energy: str,
                     exemplars: list[str] | None = None,
                     psychology: dict | None = None,
                     color: dict | None = None) -> str:
    """
    Ask Ollama to produce a short visual description for this beat.
    Psychology and color context are injected so generated scenes carry
    the correct emotional valence and color palette for the current music.
    Falls back to the beat's archetype string if Ollama is unavailable.
    """
    few_shot = ""
    if exemplars:
        examples = "\n".join(f"  - {e}" for e in exemplars[:3])
        few_shot = (
            f"\nHigh-scoring visuals for this beat (use as inspiration, not literal copy):\n"
            f"{examples}\n"
        )

    # Build psychology/color context block
    mood_block = ""
    if psychology:
        mood_label  = psychology.get("mood_label", "")
        mood_adjs   = ", ".join(psychology.get("mood_adjectives", [])[:4])
        timbre      = psychology.get("timbre_class", "")
        val         = psychology.get("valence", 0.5)
        aro         = psychology.get("arousal", 0.5)
        val_word    = "uplifting" if val > 0.55 else ("dark" if val < 0.45 else "ambivalent")
        aro_word    = "energetic" if aro > 0.55 else ("still" if aro < 0.45 else "moderate")
        tempo_label = psychology.get("tempo_label", "")
        bpm         = psychology.get("bpm", 0.0)
        bpm_line    = f"  Tempo: {int(bpm)} BPM — {tempo_label}\n" if bpm else ""
        mood_block  = (
            f"\nEmotional context (from live music analysis):\n"
            f"  Mood: {mood_label} ({mood_adjs})\n"
            f"  Tone: {val_word}, {aro_word}, timbre is {timbre}\n"
            f"{bpm_line}"
        )

    color_block = ""
    if color:
        palette   = color.get("palette_prompt", "")
        lighting  = color.get("lighting_prompt", "")
        if palette:
            color_block = (
                f"\nColor/lighting palette:\n"
                f"  Colors: {palette}\n"
                f"  Light:  {lighting}\n"
            )

    user = (
        f"Story beat: '{beat['name']}'. Audio energy: {audio_energy}.\n"
        f"Archetype: {beat['archetype']}.\n"
        f"{mood_block}"
        f"{color_block}"
        f"{few_shot}"
        f"Write ONLY a {TARGET_WORDS}-word vivid cinematic scene for Stable Diffusion. "
        f"Incorporate the mood and colors above. Output only the scene description, no preamble."
    )
    text = _ollama_api(user)
    if not text:
        text = _call_ollama(user)

    text = text.strip('"\'').split("\n")[0].strip()
    if len(text.split()) > 30:
        text = " ".join(text.split()[:25])

    return text if text else beat["archetype"]


# ── Persistent beat memory ────────────────────────────────────────────────────

def _load_beat_memory() -> dict[str, list]:
    """Load {beat_name: [(score, description), ...]} from disk."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        if os.path.exists(_MEMORY_FILE):
            with open(_MEMORY_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_beat_memory(memory: dict):
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp = _MEMORY_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(memory, f, indent=2)
        os.replace(tmp, _MEMORY_FILE)
    except Exception as e:
        print(f"[STORY] Memory save error: {e}")


# ── Main storyteller class ────────────────────────────────────────────────────

class Storyteller:
    # Keep top N high-scoring descriptions per beat as few-shot exemplars
    _EXEMPLAR_CAP  = 8
    _SCORE_THRESH  = 0.55   # min aesthetic score to qualify (capture more exemplars)
    _SAVE_INTERVAL = 10     # save memory every N score recordings

    def __init__(self):
        self._lock       = threading.Lock()
        self._beat_index = 0
        self._context    = STORY_BEATS[0]["archetype"]
        self._beat_name  = STORY_BEATS[0]["name"]
        self._audio_energy = "calm"
        self._running    = False
        self._thread     = None
        # Psychology/color context injected by the deform worker each beat
        self._psychology: dict = {}
        self._color:      dict = {}

        # Per-beat exemplar memory: {beat_name: deque[(score, description)]}
        raw = _load_beat_memory()
        self._exemplars: dict[str, deque] = {
            name: deque(pairs, maxlen=self._EXEMPLAR_CAP)
            for name, pairs in raw.items()
        }
        self._saves_pending = 0
        loaded = sum(len(v) for v in self._exemplars.values())
        if loaded:
            print(f"[STORY] Loaded {loaded} exemplars across {len(self._exemplars)} beats.")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[STORY] Started. Beat 1/8: {self._beat_name}")

    def stop(self):
        self._running = False
        _save_beat_memory(self._serialisable_exemplars())

    def set_psychology(self, psychology: dict, color: dict):
        """Update emotional context from MusicPsychologyMapper / ColorTheoryMapper."""
        self._psychology = psychology
        self._color      = color

    def set_audio_energy(self, level: str, trend: str):
        energy_map = {
            ("low",    "sustained"): "calm and meditative",
            ("low",    "rising"):    "gently building",
            ("low",    "falling"):   "softly dissolving",
            ("medium", "sustained"): "energetic and flowing",
            ("medium", "rising"):    "intensifying",
            ("medium", "falling"):   "winding down",
            ("high",   "sustained"): "explosive and overwhelming",
            ("high",   "rising"):    "approaching climax",
            ("high",   "falling"):   "cathartic release",
        }
        self._audio_energy = energy_map.get((level, trend), "dynamic")

    def get_context(self) -> str:
        with self._lock:
            return self._context

    def get_beat_name(self) -> str:
        with self._lock:
            return self._beat_name

    def record_score(self, beat_name: str, score: float):
        """
        Called by ai_worker_deform after aesthetic scoring.
        If score exceeds threshold, the current context description is stored
        as a high-quality exemplar for this beat — used as few-shot examples
        in future Ollama prompts so the storytelling improves over time.
        """
        if score < self._SCORE_THRESH:
            return
        with self._lock:
            context = self._context
        if beat_name not in self._exemplars:
            self._exemplars[beat_name] = deque(maxlen=self._EXEMPLAR_CAP)
        # Keep only high-scoring; evict lowest score when at capacity
        bucket = self._exemplars[beat_name]
        if len(bucket) >= self._EXEMPLAR_CAP:
            min_idx = min(range(len(bucket)), key=lambda i: bucket[i][0])
            items = list(bucket)
            if score > items[min_idx][0]:
                items[min_idx] = (score, context)
                self._exemplars[beat_name] = deque(items, maxlen=self._EXEMPLAR_CAP)
        else:
            bucket.append((score, context))

        self._saves_pending += 1
        if self._saves_pending >= self._SAVE_INTERVAL:
            self._saves_pending = 0
            _save_beat_memory(self._serialisable_exemplars())

    def _get_exemplars(self, beat_name: str) -> list[str]:
        """Return top-scoring descriptions for this beat (for Ollama few-shot)."""
        bucket = self._exemplars.get(beat_name)
        if not bucket:
            return []
        sorted_items = sorted(bucket, key=lambda x: -x[0])
        return [desc for _, desc in sorted_items[:3]]

    def _serialisable_exemplars(self) -> dict:
        return {name: list(dq) for name, dq in self._exemplars.items()}

    def _loop(self):
        while self._running:
            beat      = STORY_BEATS[self._beat_index]
            exemplars = self._get_exemplars(beat["name"])
            visual    = _generate_visual(
                beat, self._audio_energy, exemplars,
                psychology=self._psychology,
                color=self._color,
            )

            with self._lock:
                self._context   = visual
                self._beat_name = beat["name"]

            n_ex = len(exemplars)
            print(f"[STORY] Beat {self._beat_index + 1}/8: {beat['name']} "
                  f"({n_ex} exemplars)")
            print(f"[STORY] Context: {visual[:80]}...")

            # Beat interval driven by actual BPM: faster music = faster story progression.
            # Formula: 3600 / bpm gives seconds per beat at that tempo, scaled up so
            # the story circle completes in ~4-10 minutes depending on tempo.
            # Floor at 10s (very fast music) and ceil at 60s (very slow).
            bpm = self._psychology.get("bpm", 0.0) if self._psychology else 0.0
            if bpm > 0:
                # Each story beat lasts proportional to musical tempo
                # At 120 BPM → 30s; at 60 BPM → 60s; at 180 BPM → 20s; at 200 BPM → 18s
                interval = max(10.0, min(60.0, 3600.0 / bpm))
                # Accelerating tempo shortens the current beat (builds urgency)
                bpm_delta = self._psychology.get("bpm_delta", 0.0)
                if bpm_delta > 3.0:
                    interval *= 0.8
                elif bpm_delta < -3.0:
                    interval *= 1.2
            else:
                # Fallback to energy-based intervals when BPM not yet estimated
                energy = self._audio_energy
                if "explosive" in energy or "climax" in energy or "overwhelming" in energy:
                    interval = 15.0
                elif "calm" in energy or "meditative" in energy or "dissolving" in energy:
                    interval = 45.0
                else:
                    interval = 30.0

            time.sleep(interval)
            self._beat_index = (self._beat_index + 1) % len(STORY_BEATS)


# ── Singleton accessor ────────────────────────────────────────────────────────
_instance: Storyteller | None = None


def get_storyteller() -> Storyteller:
    global _instance
    if _instance is None:
        _instance = Storyteller()
    return _instance
