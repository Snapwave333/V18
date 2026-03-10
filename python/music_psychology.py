"""
Music Psychology Mapper — maps audio features to psychological/emotional dimensions.

Theoretical basis
-----------------
Russell's Circumplex Model of Affect (1980):
    Maps all emotions onto a 2D space of Valence × Arousal.
    - Valence  [0,1]: negative (sad/angry) ↔ positive (happy/calm)
    - Arousal  [0,1]: deactivated (sleepy/calm) ↔ activated (excited/tense)

Thayer's Two-Dimensional Mood Model:
    Energy (correlates with arousal) × Stress (inverse of valence stability).

Gabrielsson & Lindström (2001) — structural features → emotion:
    - Tempo:      fast → excited/happy/tense; slow → sad/peaceful
    - Mode:       major → happy/bright; minor → sad/dark (approximated here via
                  harmonic_ratio + centroid midrange peak)
    - Dynamics:   loud/wide range → excited/powerful; soft → tender/sad
    - Timbre:     bright centroid → alert/joyful/harsh; dark → calm/sad/mysterious
    - Rhythm:     regular (low flux) → peaceful/confident; irregular → tense/excited
    - Harmony:    high harmonic_ratio → consonant/positive; low → dissonant/tense

Eerola & Vuoskoski (2011) — spectral correlates:
    - Spectral centroid  → arousal (strongly) and valence (moderately)
    - Spectral flux       → arousal and tension
    - RMS energy         → arousal and dominance
    - ZCR (roughness)    → tension and negative valence
    - Harmonic ratio     → valence and consonance

Outputs
-------
A dict with:
    valence    [0,1]  — emotional positivity
    arousal    [0,1]  — energy/activation level
    dominance  [0,1]  — power and agency
    tension    [0,1]  — musical tension / instability
    timbre_class      — perceptual label: "warm" | "bright" | "dark" | "harsh" | "pure"
    mood_label        — discrete emotion label (for LLM prompt injection)
    mood_adjectives   — 3-5 descriptive adjectives for the current emotional state
"""

import math


# ── Timbre classification thresholds ─────────────────────────────────────────
# centroid is normalised [0,1] over 20-8000 Hz

def _timbre_class(centroid: float, harmonic_ratio: float, zcr: float) -> str:
    """
    Classify the perceptual timbre character of the current audio.

    warm   — rich, full, low-midrange dominant (centroid < 0.3, harmonic)
    bright — clear, airy, treble-forward (centroid > 0.6, harmonic)
    dark   — heavy, shadowy, sub-bass dominant (centroid < 0.25, low harm)
    harsh  — rough, abrasive, distorted (high zcr, low harmonic_ratio)
    pure   — clean, sine-like, resonant (very high harmonic_ratio)
    """
    if harmonic_ratio > 0.85:
        return "pure"
    if zcr > 0.12 and harmonic_ratio < 0.45:
        return "harsh"
    if centroid < 0.25 and harmonic_ratio < 0.55:
        return "dark"
    if centroid > 0.60 and harmonic_ratio > 0.50:
        return "bright"
    return "warm"


# ── Mood label table (Russell quadrants + dominance) ─────────────────────────
# Keyed by (valence_hi: bool, arousal_hi: bool, tension_hi: bool)
_MOOD_TABLE = {
    # high valence, high arousal
    (True,  True,  False): ("euphoric",       ["euphoric", "triumphant", "radiant", "electric"]),
    (True,  True,  True):  ("frenetic",       ["frenzied", "ecstatic", "unstoppable", "charged"]),
    # high valence, low arousal
    (True,  False, False): ("serene",         ["serene", "peaceful", "luminous", "gentle"]),
    (True,  False, True):  ("wistful",        ["wistful", "bittersweet", "tender", "nostalgic"]),
    # low valence, high arousal
    (False, True,  False): ("urgent",         ["urgent", "intense", "driving", "fierce"]),
    (False, True,  True):  ("ominous",        ["ominous", "harrowing", "volatile", "crushing"]),
    # low valence, low arousal
    (False, False, False): ("melancholic",    ["melancholic", "desolate", "somber", "hollow"]),
    (False, False, True):  ("desolate",       ["bleak", "despairing", "oppressive", "forlorn"]),
}


class MusicPsychologyMapper:
    """
    Stateless mapper: call map(features_dict) every frame to get psychological context.

    Uses a lightweight EMA to smooth the psychological dimensions over time,
    preventing rapid flickering between mood states.
    """

    _EMA = 0.18   # smoothing alpha for psychological dimensions

    def __init__(self):
        self._val  = 0.5
        self._aro  = 0.5
        self._dom  = 0.5
        self._ten  = 0.5

    def map(self, features: dict) -> dict:
        """
        Parameters
        ----------
        features : dict from AudioFeatures.to_dict()

        Returns
        -------
        dict with keys: valence, arousal, dominance, tension,
                        timbre_class, mood_label, mood_adjectives
        """
        # ── Extract features (with safe defaults) ────────────────────────────
        rms       = float(features.get("smoothed_rms",    0.0))
        bass      = float(features.get("bass",            0.0))
        mid       = float(features.get("mid",             0.0))
        high      = float(features.get("high",            0.0))
        centroid  = float(features.get("centroid",        0.3))
        bpm       = float(features.get("bpm",             120.0))
        bpm_delta = float(features.get("bpm_delta",       0.0))
        beat_str  = float(features.get("beat_strength",   0.0))
        flux      = float(features.get("spectral_flux",   0.0))
        rolloff   = float(features.get("spectral_rolloff",2000.0))
        zcr       = float(features.get("zcr",             0.05))
        harm      = float(features.get("harmonic_ratio",  0.5))
        dynrange  = float(features.get("dynamic_range",   1.0))

        # ── Normalize to [0,1] ────────────────────────────────────────────────
        rms_n    = min(rms / 5000.0, 1.0)
        bpm_n    = max(0.0, min((bpm - 60.0) / 140.0, 1.0))   # 60→0, 200→1
        rolloff_n = min(rolloff / 8000.0, 1.0)
        dynr_n   = min((dynrange - 0.5) / 4.5, 1.0)           # 0.5→0, 5.0→1
        bass_dominance = min(bass / (mid + high + 1.0), 1.0)

        # ── Arousal (energy/activation) ───────────────────────────────────────
        # Strongly driven by tempo, loudness, brightness, and flux.
        # Research weights: bpm 30%, rms 30%, centroid 20%, flux 20%
        arousal_raw = (
            0.30 * bpm_n
            + 0.30 * rms_n
            + 0.20 * centroid
            + 0.20 * flux
        )
        arousal_raw = float(min(max(arousal_raw, 0.0), 1.0))

        # ── Valence (positivity) ──────────────────────────────────────────────
        # Estimated from:
        #   harmonic_ratio  → consonance → positive valence
        #   centroid peak   → midrange (0.35) = warmest; extremes = harsher
        #   bass dominance  → dark bass = lower valence
        #   zcr roughness   → high roughness = lower valence
        centroid_harmony = 1.0 - abs(centroid - 0.35) / 0.65   # peak at 0.35
        valence_raw = (
            0.38 * harm
            + 0.28 * centroid_harmony
            + 0.22 * (1.0 - bass_dominance)
            + 0.12 * (1.0 - min(zcr / 0.15, 1.0))
        )
        valence_raw = float(min(max(valence_raw, 0.0), 1.0))

        # ── Dominance (power/agency) ──────────────────────────────────────────
        # High bass + loud + strong beats → dominant
        dominance_raw = (
            0.40 * bass_dominance
            + 0.35 * rms_n
            + 0.25 * beat_str
        )
        dominance_raw = float(min(max(dominance_raw, 0.0), 1.0))

        # ── BPM tempo label and acceleration signal ───────────────────────────
        # bpm_delta: positive = speeding up (more tension), negative = slowing (release)
        # Clamp to ±20 BPM/s for normalization (typical accelerando range)
        bpm_accel = float(min(max(bpm_delta / 20.0, -1.0), 1.0))  # [-1, 1]
        # Map to [0,1]: speeding up → higher tension, slowing → lower
        bpm_tension = (bpm_accel + 1.0) / 2.0

        # ── Tension (instability/dissonance) ──────────────────────────────────
        # High flux, low harmonic ratio, compressed dynamics, irregular rhythm,
        # and tempo acceleration all contribute to tension.
        tension_raw = (
            0.30 * flux
            + 0.25 * (1.0 - harm)
            + 0.20 * (1.0 - dynr_n)
            + 0.15 * min(zcr / 0.15, 1.0)
            + 0.10 * bpm_tension
        )
        tension_raw = float(min(max(tension_raw, 0.0), 1.0))

        # ── EMA smoothing — prevent mood flickering ───────────────────────────
        a = self._EMA
        self._val = (1 - a) * self._val + a * valence_raw
        self._aro = (1 - a) * self._aro + a * arousal_raw
        self._dom = (1 - a) * self._dom + a * dominance_raw
        self._ten = (1 - a) * self._ten + a * tension_raw

        # ── Timbre classification ─────────────────────────────────────────────
        t_class = _timbre_class(centroid, harm, zcr)

        # ── Discrete mood label ───────────────────────────────────────────────
        key = (self._val > 0.50, self._aro > 0.50, self._ten > 0.55)
        mood_label, mood_adjectives = _MOOD_TABLE.get(key, ("dynamic", ["dynamic", "shifting", "evocative"]))

        # ── Tempo label for Ollama prompt injection ───────────────────────────
        if bpm < 80:
            tempo_label = "slow, meditative"
        elif bpm < 110:
            tempo_label = "moderate, flowing"
        elif bpm < 140:
            tempo_label = "driving, energetic"
        elif bpm < 170:
            tempo_label = "fast, intense"
        else:
            tempo_label = "relentless, frantic"

        if bpm_delta > 3.0:
            tempo_label += ", accelerating"
        elif bpm_delta < -3.0:
            tempo_label += ", decelerating"

        return {
            "valence":         round(self._val, 3),
            "arousal":         round(self._aro, 3),
            "dominance":       round(self._dom, 3),
            "tension":         round(self._ten, 3),
            "timbre_class":    t_class,
            "mood_label":      mood_label,
            "mood_adjectives": mood_adjectives,
            "bpm":             round(bpm, 1),
            "bpm_delta":       round(bpm_delta, 2),
            "tempo_label":     tempo_label,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: MusicPsychologyMapper | None = None


def get_mapper() -> MusicPsychologyMapper:
    global _instance
    if _instance is None:
        _instance = MusicPsychologyMapper()
    return _instance
