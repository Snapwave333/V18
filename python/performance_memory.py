"""
Performance Memory — persistent ML learning store.

Tracks which prompt elements (subjects, descriptors, styles, story contexts)
produced aesthetically high-scoring frames. Persists between sessions.

Learning mechanism:
    1. Susa selects a subject/descriptor/style and generates a prompt.
    2. Generated frame is scored by AestheticScorer (CLIP aesthetic quality).
    3. Score is recorded here via EMA per token.
    4. Susa queries quality_weight(token) which returns [0.1, 2.0].
    5. High-scoring tokens get selection weight boost → used more often.
    6. Over time Susa converges toward the visually best elements.

This is a gradient-free online bandit learning approach: no training loop,
no backprop — just weighted random selection guided by reward signal.
"""

import json
import os
import threading
import time

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
SAVE_FILE = os.path.join(DATA_DIR, "performance_memory.json")

EMA_ALPHA   = 0.10   # learning rate — slightly smoother than 0.12
MIN_SAMPLES = 3      # start influencing after 3 observations (faster signal)
SAVE_EVERY  = 30     # save to disk every N record() calls
BOOST_MAX   = 2.0    # max selection weight multiplier
BOOST_MIN   = 0.15   # min selection weight multiplier
DECAY_RATE  = 0.002  # per-record nudge toward neutral (prevents stale dominance)


class PerformanceMemory:
    """
    Thread-safe persistent EMA score tracker for prompt tokens.

    Each 'token' is any string element used in prompt generation:
    a subject phrase, a descriptor, a style, a story context snippet.

    quality_weight(token) returns a float that multiplies the token's
    base selection probability in Susa's weighted_choice.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._scores: dict[str, float] = {}   # token → EMA score
        self._counts: dict[str, int]   = {}   # token → observation count
        self._pending = 0
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, tokens: list[str], score: float):
        """
        Record an aesthetic score against a list of tokens.
        Call after every scored frame.

        Parameters
        ----------
        tokens : list of string elements that produced the scored frame
        score  : float [0, 1] from AestheticScorer
        """
        with self._lock:
            for token in tokens:
                if token not in self._scores:
                    self._scores[token] = score
                    self._counts[token] = 1
                else:
                    n = self._counts[token]
                    # Adaptive alpha: trust new data more when we have few samples
                    alpha = max(EMA_ALPHA, 1.0 / (n + 1))
                    self._scores[token] = (
                        (1 - alpha) * self._scores[token] + alpha * score
                    )
                    self._counts[token] = min(n + 1, 9999)

            # Score decay — gently nudge all tracked scores toward neutral (0.5)
            # so stale high/low scores don't lock in forever as tastes evolve.
            for token in self._scores:
                if self._counts.get(token, 0) >= MIN_SAMPLES:
                    self._scores[token] += DECAY_RATE * (0.5 - self._scores[token])

            self._pending += 1
            if self._pending >= SAVE_EVERY:
                self._pending = 0
                self._save()

    def quality_weight(self, token: str) -> float:
        """
        Returns selection weight multiplier for a token.

        Returns
        -------
        float in [BOOST_MIN, BOOST_MAX]
            1.0 = neutral (no data or average score)
            >1.0 = this token has produced good-looking frames
            <1.0 = this token has produced poor-looking frames
        """
        with self._lock:
            if self._counts.get(token, 0) < MIN_SAMPLES:
                return 1.0
            score = self._scores[token]
        # Map [0, 1] score → [BOOST_MIN, BOOST_MAX] weight
        # score=0.5 → weight=1.0 (neutral midpoint)
        weight = BOOST_MIN + score * (BOOST_MAX - BOOST_MIN)
        return float(weight)

    def top_tokens(self, n: int = 15) -> list[tuple[str, float, int]]:
        """Return top N highest-scoring tokens for debugging/monitoring."""
        with self._lock:
            qualified = [
                (t, self._scores[t], self._counts[t])
                for t in self._scores
                if self._counts.get(t, 0) >= MIN_SAMPLES
            ]
        return sorted(qualified, key=lambda x: -x[1])[:n]

    def bottom_tokens(self, n: int = 10) -> list[tuple[str, float, int]]:
        """Return bottom N lowest-scoring tokens."""
        with self._lock:
            qualified = [
                (t, self._scores[t], self._counts[t])
                for t in self._scores
                if self._counts.get(t, 0) >= MIN_SAMPLES
            ]
        return sorted(qualified, key=lambda x: x[1])[:n]

    def summary(self) -> str:
        with self._lock:
            total    = len(self._scores)
            qualified = sum(1 for t in self._scores if self._counts.get(t, 0) >= MIN_SAMPLES)
            if not self._scores:
                return "PerformanceMemory: empty"
            avg = sum(self._scores.values()) / len(self._scores)
        return (
            f"PerformanceMemory: {total} tokens ({qualified} qualified) "
            f"avg_score={avg:.3f}"
        )

    def flush(self):
        """Force immediate disk save."""
        self._save()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            if not os.path.exists(SAVE_FILE):
                print("[MEMORY] No performance data yet — starting fresh.")
                return
            with open(SAVE_FILE) as f:
                data = json.load(f)
            self._scores = data.get("scores", {})
            self._counts = data.get("counts", {})
            total = len(self._scores)
            qualified = sum(1 for t in self._scores if self._counts.get(t, 0) >= MIN_SAMPLES)
            print(
                f"[MEMORY] Loaded {total} tokens ({qualified} qualified, "
                f"saved {data.get('saved_at', 'unknown time')[:19]})."
            )
        except Exception as e:
            print(f"[MEMORY] Load error: {e}. Starting fresh.")

    def _save(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = SAVE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "scores":   self._scores,
                        "counts":   self._counts,
                        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "version":  1,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp, SAVE_FILE)
        except Exception as e:
            print(f"[MEMORY] Save error: {e}")


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: PerformanceMemory | None = None


def get_memory() -> PerformanceMemory:
    global _instance
    if _instance is None:
        _instance = PerformanceMemory()
    return _instance
