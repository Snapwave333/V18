"""
Susa — Intelligent Prompt Generation Engine for Vibes VJ

Maintains a weighted memory database so prompts never feel repetitive.
Uses usage-decay scoring, thematic drift detection, intensity trend analysis,
and structural variety enforcement to produce an ever-evolving narrative arc.
"""
import random
import time
import math
from collections import deque

# ML performance memory — learns which elements look best over time
from performance_memory import get_memory as _get_memory

# ---------------------------------------------------------------------------
# Quality suffix and negative prompt (imported by ai_generator.py)
# ---------------------------------------------------------------------------
_QUALITY_SUFFIX = (
    "highly realistic, photorealistic masterpiece, 8k resolution, cinematic lighting, sharp details, vibrant colors, rich textures, professional photography"
)

NEG_EXTRA = "abstract, conceptual, surreal, weird, low detail, smudged, blurry, lowres, text, watermark, signature, error, cropped, worst quality, low quality, normal quality, jpeg artifacts, duplicate, out of focus, monochrome, grayscale, desaturated, dull, muddy colors, grey, brown, beige"
NEGATIVE_PROMPT = (
    f"kaleidoscope, fractal, geometric shapes, psychedelic swirls, mandala, spirograph, procedural noise, trippy, symmetric, tessellation, op art, generative art, glitch art, {NEG_EXTRA}"
)

# ---------------------------------------------------------------------------
# Word banks — large enough that weight-decay keeps variety high
# ---------------------------------------------------------------------------
_BANKS = {
    "low": {
        "subjects": [
            "A lone traveler resting under a twisted oak tree at twilight",
            "An abandoned gas station in the middle of a vast desert, cinematic wide shot",
            "A peaceful, sunlit clearing in a dense ancient forest, mossy rocks",
            "A solitary figure standing on a misty, rocky shoreline",
            "An empty, softly lit subway car traveling late at night",
            "A small cabin glowing warmly in a snow-covered valley",
            "A dusty, forgotten library filled with towering bookshelves",
            "A quiet alleyway illuminated by a single flickering street lamp",
            "A silhouette walking through a field of tall grass at golden hour",
            "An old wooden boat tethered to a dock on a perfectly still lake",
            "A monk meditating in a shadowy, incense-filled stone temple",
            "A close-up of a weathered map and a compass resting on a wooden desk",
            "A vast, empty hallway in an ornate, crumbling mansion",
            "A single glowing lantern sitting on a mossy stone wall",
            "A distant figure walking into a valley shrouded by heavy fog",
        ],
        "descriptors": [
            "Quiet and atmospheric, building subtle tension",
            "Calm and introspective, focused on natural beauty",
            "Serene and mysterious, deeply shadowed",
            "Peaceful yet melancholic, isolated",
            "Still and grounded, highly detailed",
            "Softly illuminated, inviting yet lonely",
            "Minimalist and moody, drenched in atmosphere",
            "Ancient and timeless, weathered by history",
            "Tender and emotional, focusing on solitude",
        ],
        "styles": [
            "Cinematic landscape photography, Arri Alexa 65, desaturated natural tones",
            "Moody noir cinematography, deep shadows, soft volumetric lighting",
            "Golden hour landscape, wide-angle lens, warm natural light",
            "Documentary style realism, 35mm film grain, muted color palette",
            "Atmospheric environmental portrait, shallow depth of field, cool tones",
            "Gothic realism, high contrast, stark lighting",
            "Quiet indie film aesthetic, pastel shadows, soft focus",
            "National Geographic style photography, hyper-detailed, naturalistic",
            "Ethereal realism, mist, diffuse lighting, cinematic aspect ratio",
        ],
    },
    "medium": {
        "subjects": [
            "A bustling futuristic city market with neon-lit food stalls and cinematic rain",
            "A sleek electric supercar racing through a desert at sunset, motion blur",
            "A highly detailed sci-fi laboratory with glowing holographic displays",
            "A dramatic mountain monastery hanging over a deep misty gorge",
            "A crowded street in Tokyo at night, vibrant neon signs reflecting on wet asphalt",
            "A high-tech control room with massive screens showing galactic maps",
            "A massive waterfall crashing into a tropical turquoise lagoon",
            "A futuristic space station interior with a view of a swirling nebula",
            "A detailed medieval armor suit reflecting a roaring fireplace",
            "A hidden canyon with bioluminescent plants glowing in the darkness",
        ],
        "descriptors": [
            "Dynamic and vibrant, capturing a moment of high energy",
            "Detailed and crisp, with rich textures and bold lighting",
            "Cinematic and grand, showing immense scale and depth",
            "Vivid and alive, drenched in saturated colors",
            "Intense and sharp, focusing on realistic mechanics and light",
        ],
        "styles": [
            "Modern cinematic photography, vivid color grade, high dynamic range",
            "Cyberpunk realism, rich color reflections, sharp digital focus",
            "National Geographic style, vibrant natural lighting, hyper-detailed textures",
            "Commercial car photography, sleek lighting, deep glossy reflections",
            "Sci-fi film aesthetic, crisp anamorphic flares, teal and orange vibrancy",
        ],
    },
    "high": {
        "subjects": [
            "A colossal explosion destroying a skyscraper, debris flying toward the camera",
            "A mythical dragon breathing fire over a massive, burning castle",
            "A climactic battle between armies colliding on a muddy, blood-soaked field",
            "A spaceship crashing violently into the surface of an alien planet",
            "Reality fracturing during a magical duel, pure energy tearing the scenery apart",
            "A tidal wave of biblical proportions crashing down on a modern city",
            "A superhero unleashing maximum power in a cratered, destroyed city square",
            "A portal opening in the sky, raining fire and lightning down on soldiers",
            "A monstrous leviathan rising from a stormy ocean, lightning striking",
            "A volcano erupting violently, raining lava and ash over a fleeing village",
            "A catastrophic train derailment exploding on a bridge, fire and smoke",
            "An intense, close-quarters sword fight, sparks flying, faces grimacing in agony",
            "A massive orbital laser striking a planetary shield, blinding light",
            "A chaotic riot breaking through the barricades of a futuristic citadel",
            "A wizard summoning a devastating meteor swarm, sky burning red and gold",
        ],
        "descriptors": [
            "Violently explosive and overwhelming, apocalyptic scale",
            "Chaotic and destructive, maximum visual intensity",
            "Climactic and devastating, high-stakes final battle",
            "Unstable and earth-shattering, raw power unleashed",
            "Frenzied and panicked, portraying total destruction",
            "Epic and cataclysmic, a world-ending event",
            "Blindingly intense action, raw visceral energy",
            "At the emotional breaking point, pure adrenaline and chaos",
        ],
        "styles": [
            "Explosive action cinematography, high-speed capture, Michael Bay style",
            "Disaster movie realism, extreme destruction, smoke and fire grading",
            "Dark fantasy climax, apocalyptic sky, heavy HDR contrast",
            "Sci-fi battlefield, blinding laser light, gritty desaturation",
            "Kaiju movie framing, ground-level tilted angle, massive scale",
            "War epic, slow-motion mud and blood, visceral 35mm film",
            "High-adrenaline thriller, motion blur, harsh strobe lighting",
            "Apocalyptic landscape, blinding white light core, deep dramatic shadows",
            "Fantasy magic battle, VFX overload, highly saturated spell effects",
        ],
    },
}

# Sentence structure templates — {s}=subject, {d}=descriptor, {st}=style
_STRUCTURES = [
    "{s}, {d}, {st}",
    "{d} {s}, {st}",
    "{s} — {st}, {d}",
    "A {d} scene: {s}, {st}",
    "{s}; {st}; {d}",
    "{st} depiction of {s}, {d}",
    "{d} visualization: {s}, rendered in {st}",
]


# ---------------------------------------------------------------------------
# Usage memory: tracks per-token usage with time-based decay
# ---------------------------------------------------------------------------
class _UsageMemory:
    """
    Tracks how recently each token (subject/descriptor/style/structure)
    was used. Returns a weight in (0, 1] — recently used tokens get low weight.
    Decay half-life is configurable in seconds.
    """
    HALF_LIFE = 120.0  # seconds until weight recovers to 50%

    def __init__(self):
        # token -> timestamp of last use
        self._last_used: dict[str, float] = {}

    def weight(self, token: str) -> float:
        if token not in self._last_used:
            return 1.0
        elapsed = time.time() - self._last_used[token]
        # Exponential decay: w = 1 - exp(-elapsed / halflife * ln2)
        return 1.0 - math.exp(-elapsed / self.HALF_LIFE * math.log(2))

    def mark_used(self, token: str):
        self._last_used[token] = time.time()

    def weighted_choice(self, tokens: list[str]) -> str:
        weights = [max(self.weight(t), 0.05) for t in tokens]
        return random.choices(tokens, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Intensity trend tracker
# ---------------------------------------------------------------------------
class _IntensityTracker:
    """
    Maintains a rolling window of intensity values and computes:
    - current level (low / medium / high)
    - trend (rising / falling / sustained)
    """
    def __init__(self, window=30):
        self._history: deque[float] = deque(maxlen=window)

    def update(self, intensity: float):
        self._history.append(intensity)

    def level(self) -> str:
        if not self._history:
            return "low"
        avg = sum(self._history) / len(self._history)
        if avg < 500:
            return "low"
        elif avg < 4000:
            return "medium"
        return "high"

    def trend(self) -> str:
        """Returns 'rising', 'falling', or 'sustained'."""
        h = list(self._history)
        if len(h) < 6:
            return "sustained"
        first_half = sum(h[:len(h)//2]) / (len(h)//2)
        second_half = sum(h[len(h)//2:]) / (len(h) - len(h)//2)
        delta = second_half - first_half
        if delta > 200:
            return "rising"
        elif delta < -200:
            return "falling"
        return "sustained"


# ---------------------------------------------------------------------------
# Thematic drift detector
# ---------------------------------------------------------------------------
class _ThemeMemory:
    """
    Tracks high-level 'themes' of recent prompts (the subject category).
    Penalizes subjects that share keywords with recently used subjects
    so the imagery drifts meaningfully rather than cycling.
    """
    def __init__(self, window=12):
        self._recent: deque[str] = deque(maxlen=window)

    def record(self, subject: str):
        # Extract meaningful words (len > 4) as theme fingerprint
        words = set(w.lower() for w in subject.split() if len(w) > 4)
        self._recent.append(" ".join(sorted(words)))

    def novelty_weight(self, subject: str) -> float:
        """Lower weight if subject shares keywords with recent history."""
        if not self._recent:
            return 1.0
        words = set(w.lower() for w in subject.split() if len(w) > 4)
        max_overlap = 0
        for past in self._recent:
            past_words = set(past.split())
            overlap = len(words & past_words) / max(len(words | past_words), 1)
            max_overlap = max(max_overlap, overlap)
        return max(1.0 - max_overlap * 1.5, 0.05)


# ---------------------------------------------------------------------------
# Main Susa class
# ---------------------------------------------------------------------------
class Susa:
    def __init__(self):
        self._usage = _UsageMemory()
        self._intensity = _IntensityTracker()
        self._themes = _ThemeMemory()
        self._structure_usage = _UsageMemory()
        self._last_level = "low"
        self._call_count = 0
        self._last_fx_time = 0.0
        # ML learning: persistent quality memory
        self._memory = _get_memory()
        # Track tokens used in last generate_prompt call for scoring callback
        self.last_tokens: list[str] = []
        # Prompt inertia: EMA of recent aesthetic scores for current prompt
        # Prompt inertia: if the current prompt is scoring well (>0.62) and
        # the audio intensity level hasn't changed, keep it rather than re-rolling.
        self._prompt_score_ema: float = 0.5
        self._last_prompt_tokens: list[str] = []
        self._last_prompt_result: str = ""
        # Narrative blending
        self._last_narrative: str = ""
        self._narrative_transition_counter: int = 0
        self._TRANSITION_DURATION = 3 # number of prompt cycles to blend

    def record_aesthetic(self, tokens: list[str], score: float):
        """
        Called by the AI worker after an aesthetic scorer rates a frame.
        Updates the performance memory so high-scoring elements get selected more.
        Also updates prompt inertia EMA.
        """
        self._memory.record(tokens, score)
        # Update inertia EMA (fast alpha so inertia reacts within a few frames)
        self._prompt_score_ema = 0.25 * score + 0.75 * self._prompt_score_ema

    def _ml_weight(self, token: str) -> float:
        """Multiply time-decay weight by ML quality weight."""
        return self._memory.quality_weight(token)

    def generate_prompt(self, features: dict, narrative: str = "",
                        psychology: dict | None = None,
                        color: dict | None = None) -> str:
        """
        features   : dict from AudioFeatures.to_dict() — or a plain float.
        narrative  : story-circle visual context from Storyteller.
        psychology : dict from MusicPsychologyMapper.map() — adds emotional depth.
        color      : dict from ColorTheoryMapper.map() — adds color/lighting context.
        """
        self._call_count += 1

        # Backward compat: accept bare float
        if isinstance(features, (int, float)):
            features = {"smoothed_rms": float(features)}

        # Prompt inertia: if the current prompt is scoring well (>0.62) and
        # the audio intensity level hasn't changed, keep it rather than re-rolling.
        # This lets a visually good moment breathe instead of constantly cycling.
        if (
            self._last_prompt_result
            and self._prompt_score_ema > 0.62
            and self._intensity.level() == self._last_level
            and random.random() < 0.55   # still allow change 45% of the time
        ):
            self.last_tokens = list(self._last_prompt_tokens)
            return self._last_prompt_result

        rms       = float(features.get("smoothed_rms", 0.0))
        sub_bass  = float(features.get("sub_bass", 0.0))
        bass      = float(features.get("bass", 0.0))
        mid       = float(features.get("mid", 0.0))
        high      = float(features.get("high", 0.0))
        centroid  = float(features.get("centroid", 0.0))   # 0=dark/warm, 1=bright/airy
        beat      = bool(features.get("beat", False))
        transient = bool(features.get("transient", False))
        beat_str  = float(features.get("beat_strength", 0.0))

        self._intensity.update(rms)
        level = self._intensity.level()
        trend = self._intensity.trend()
        bank  = _BANKS[level]

        # ── Subject — time-decay × thematic novelty × ML quality weight ────────
        subjects = bank["subjects"]
        subject_weights = [
            self._usage.weight(s) * self._themes.novelty_weight(s) * self._ml_weight(s)
            for s in subjects
        ]
        subject = random.choices(subjects, weights=[max(w, 0.02) for w in subject_weights], k=1)[0]

        # Intensity trend bridges
        if trend == "rising" and level == "low" and random.random() < 0.3:
            subject = self._usage.weighted_choice(_BANKS["medium"]["subjects"])
        elif trend == "falling" and level == "high" and random.random() < 0.3:
            subject = self._usage.weighted_choice(_BANKS["medium"]["subjects"])

        # Beat/transient → visual punctuation: jump to adjacent bank
        if (beat or transient) and beat_str > 0.5 and random.random() < 0.4:
            if level == "low":
                subject = self._usage.weighted_choice(_BANKS["medium"]["subjects"])
            elif level == "medium":
                subject = self._usage.weighted_choice(_BANKS["high"]["subjects"])

        # ── Descriptor — with ML quality weight ───────────────────────────────
        descriptors = list(bank["descriptors"])
        if centroid > 0.6 and level != "high":
            descriptors += _BANKS["medium"]["descriptors"][:3]
        elif centroid < 0.2 and sub_bass > bass:
            descriptors += _BANKS["high"]["descriptors"][:2]

        desc_weights = [
            self._usage.weight(d) * self._ml_weight(d) for d in descriptors
        ]
        descriptor = random.choices(descriptors, weights=[max(w, 0.02) for w in desc_weights], k=1)[0]

        # ── Style — with ML quality weight ────────────────────────────────────
        styles = list(bank["styles"])
        if mid > bass * 1.5 and level == "low":
            styles += ["golden hour portrait photography, warm natural light, shallow DOF",
                       "soft studio lighting, rich skin tones, creamy bokeh"]
        if sub_bass > mid * 2.0:
            styles += ["dark moody concert photography, deep blacks, selective warm highlights",
                       "chiaroscuro studio photography, dramatic shadow play, rich tones"]
        if high > mid * 1.8:
            styles += ["high-key studio photography, clean white light, sharp detail",
                       "macro photography with ring flash, vivid natural colors, extreme detail"]

        style_weights = [
            self._usage.weight(s) * self._ml_weight(s) for s in styles
        ]
        style = random.choices(styles, weights=[max(w, 0.02) for w in style_weights], k=1)[0]

        # ── Structure — with ML quality weight ────────────────────────────────
        struct_weights = [
            self._structure_usage.weight(s) * self._ml_weight(s) for s in _STRUCTURES
        ]
        structure = random.choices(_STRUCTURES, weights=[max(w, 0.02) for w in struct_weights], k=1)[0]

        # ── Psychology-driven descriptor augmentation ─────────────────────────
        # Inject mood adjectives from the music psychology mapper so the
        # descriptor layer carries emotional truth derived from the actual audio.
        psych_clause = ""
        if psychology:
            mood_adjs = psychology.get("mood_adjectives", [])
            timbre    = psychology.get("timbre_class", "")
            if mood_adjs:
                psych_clause = ", ".join(mood_adjs[:3])  # top 3 adjectives
            if timbre and timbre not in ("warm",):        # warm is the default, skip it
                psych_clause = f"{timbre}-timbred, {psych_clause}" if psych_clause else timbre

        # ── Color/lighting injection ──────────────────────────────────────────
        # Use palette_prompt only (not full_color_context) to stay within
        # SD 1.5's 77-token CLIP limit.
        color_clause = ""
        if color:
            color_clause = color.get("palette_prompt", "")

        # ── Narrative blending ───────────────────────────────────────────────
        if narrative != self._last_narrative:
            # Start a transition if the story beat changed
            if self._last_narrative:
                self._narrative_transition_counter = self._TRANSITION_DURATION
            self._last_narrative = narrative

        if self._narrative_transition_counter > 0:
            # Blend the new narrative with the old one
            old_part = " ".join(self._last_narrative.split()[:8])
            new_part = narrative
            effective_subject = f"{new_part}, transitioning from the previous scene of {old_part}"
            self._narrative_transition_counter -= 1
        else:
            effective_subject = narrative if narrative else subject

        core = (
            structure
            .replace("{s}", effective_subject)
            .replace("{d}", descriptor)
            .replace("{st}", style)
        )

        # Mark used (time-decay memory)
        self._usage.mark_used(subject)
        self._usage.mark_used(descriptor)
        self._usage.mark_used(style)
        self._structure_usage.mark_used(structure)
        self._themes.record(effective_subject)
        self._last_level = level

        # Expose tokens for aesthetic feedback loop (include structure so it learns too)
        self.last_tokens = [subject, descriptor, style, structure]
        if narrative:
            self.last_tokens.append(narrative[:80])   # narrative key (truncated)

        # Assemble final prompt — psychology/color appended after core content
        # so they refine but don't override the subject/narrative.
        # Keep total well within SD 1.5's 77-token limit.
        suffix_parts = [_QUALITY_SUFFIX]
        if psych_clause:
            suffix_parts.append(psych_clause)
        if color_clause:
            suffix_parts.append(color_clause)

        result = f"{core}, {', '.join(suffix_parts)}"
        self._last_prompt_result = result
        self._last_prompt_tokens = list(self.last_tokens)
        return result

    def generate_fx_state(self, features: dict, beat_name: str = "") -> dict:
        """
        Generate FX directives — reserved, musical, intentional.
        Only triggers dramatic FX on story-aligned moments with strong audio.
        Returns a dict matching the vj_fx_state.json schema.
        """
        state = {
            "fx_mode": 0,
            "fx_intensity": 0.0,
            "hue_shift": 0.0,
            "saturation_boost": 0.0,
            "warp_amount": 0.0,
            "glow_strength": 0.0,
            "ascii_force": 1.0,   # ASCII always on — it's the focal point
            "scanline_strength": 0.0,
        }

        if isinstance(features, (int, float)):
            features = {"smoothed_rms": float(features)}

        rms       = float(features.get("smoothed_rms", 0.0))
        bass      = float(features.get("bass", 0.0))
        high      = float(features.get("high", 0.0))
        beat_str  = float(features.get("beat_strength", 0.0))
        transient = bool(features.get("transient", False))
        centroid  = float(features.get("centroid", 0.0))

        level = self._intensity.level()
        trend = self._intensity.trend()

        # ── Continuous subtle modulation (always active) ─────────────────────
        state["hue_shift"]        = (random.random() - 0.5) * 0.15
        state["saturation_boost"] = min(rms / 5000.0 * 0.25, 0.25)
        state["warp_amount"]      = min(bass / 2000.0 * 0.25, 0.25)
        state["glow_strength"]    = min(rms / 3000.0 * 0.4, 0.5)
        state["scanline_strength"] = 0.08 if transient else 0.0

        # Larger hue drift on story-beat transitions
        if beat_name and "GO" in beat_name.upper():
            state["hue_shift"] = (random.random() - 0.5) * 0.4
        elif beat_name and "CHANGE" in beat_name.upper():
            state["hue_shift"] = (random.random() - 0.5) * 0.6

        # Saturation follows energy trend
        if trend == "rising":
            state["saturation_boost"] += 0.1
        elif trend == "falling":
            state["saturation_boost"] -= 0.1

        # ── Dramatic FX triggering — RESERVED ────────────────────────────────
        now = time.time()
        cooldown_ok = (now - self._last_fx_time) > 30.0

        if not cooldown_ok:
            return state

        # Story-beat alignment
        dramatic_beats = {"FIND", "TAKE", "CHANGE"}
        on_dramatic_beat = any(b in beat_name.upper() for b in dramatic_beats) if beat_name else False

        # Kaleidoscope on dramatic beats + heavy bass
        if on_dramatic_beat and beat_str > 0.7 and bass > 1200 and random.random() < 0.15:
            state["fx_mode"] = 3
            state["fx_intensity"] = min(beat_str * 0.8, 0.85)
            self._last_fx_time = now
            print(f"[AI FX] mode=kaleido intensity={state['fx_intensity']:.2f} reason='{beat_name} + bass drop'")

        # Mirror on sustained high energy
        elif level == "high" and trend == "sustained" and beat_str > 0.6 and random.random() < 0.12:
            state["fx_mode"] = 1
            state["fx_intensity"] = 0.6
            self._last_fx_time = now
            print(f"[AI FX] mode=mirror intensity=0.60 reason='sustained high energy'")

        # Quad on drops (falling from high)
        elif level == "high" and trend == "falling" and random.random() < 0.10:
            state["fx_mode"] = 2
            state["fx_intensity"] = 0.65
            self._last_fx_time = now
            print(f"[AI FX] mode=quad intensity=0.65 reason='cathartic drop'")

        # Edge glow on treble spikes
        elif high > 1000 and centroid > 0.7 and transient and random.random() < 0.08:
            state["fx_mode"] = 4
            state["fx_intensity"] = 0.5
            self._last_fx_time = now
            print(f"[AI FX] mode=edge intensity=0.50 reason='treble transient spike'")

        return state


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    susa = Susa()
    test_cases = [
        {"smoothed_rms": 50,   "centroid": 0.1, "beat": False, "transient": False, "beat_strength": 0.1, "sub_bass": 200, "bass": 100, "mid": 50,   "high": 20},
        {"smoothed_rms": 300,  "centroid": 0.4, "beat": True,  "transient": False, "beat_strength": 0.6, "sub_bass": 400, "bass": 300, "mid": 200,  "high": 100},
        {"smoothed_rms": 1500, "centroid": 0.6, "beat": True,  "transient": True,  "beat_strength": 0.9, "sub_bass": 800, "bass": 700, "mid": 900,  "high": 600},
        {"smoothed_rms": 5000, "centroid": 0.8, "beat": True,  "transient": True,  "beat_strength": 1.0, "sub_bass": 900, "bass": 800, "mid": 1200, "high": 1100},
        {"smoothed_rms": 200,  "centroid": 0.2, "beat": False, "transient": False, "beat_strength": 0.2, "sub_bass": 500, "bass": 200, "mid": 80,   "high": 10},
    ]
    print("--- Susa prompt generation test ---\n")
    for i, feat in enumerate(test_cases * 3):
        prompt = susa.generate_prompt(feat)
        print(f"[{i:02d}] rms={feat['smoothed_rms']:5.0f}  level={susa._intensity.level():6s}  trend={susa._intensity.trend():9s}")
        print(f"     {prompt[:100]}\n")
        print(f"     {prompt}\n")
