"""
AudioIngest — Full spectral audio analysis engine.

Extracts per-frame:
  - rms           : overall loudness (RMS, 16-bit scale)
  - sub_bass      : 20-80 Hz energy   (kick drum body, deep rumble)
  - bass          : 80-250 Hz energy  (bass guitar, kick attack)
  - mid           : 250-2000 Hz energy (vocals, guitars, melodic content)
  - high          : 2000-8000 Hz energy (hi-hats, brightness, air)
  - centroid      : spectral centroid normalised to [0,1] (brightness proxy)
  - beat          : True if a beat onset was detected this frame
  - transient     : True if a sharp transient (drum hit) was detected
  - beat_strength : smoothed onset strength [0,1]
  - smoothed_rms  : rolling-average rms for stable prompt decisions
  - spectral_flux : half-wave rectified spectral flux (onset energy change rate)
  - spectral_rolloff : frequency (Hz) below which 85% of energy lies (brightness/richness)
  - zcr           : zero crossing rate [0,1] — roughness/noisiness of signal
  - harmonic_ratio: spectral stability [0,1] — 1=tonal/harmonic, 0=noisy/percussive
  - dynamic_range : short-to-long RMS ratio — expressive dynamics vs compressed
"""

import numpy as np
import pyaudio
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
RATE  = 44100
CHUNK = 2048   # ~46 ms per frame — good latency/resolution tradeoff

BANDS = {
    "sub_bass": (20,   80),
    "bass":     (80,   250),
    "mid":      (250,  2000),
    "high":     (2000, 8000),
}

RMS_SMOOTH      = 20
BAND_SMOOTH     = 8
ONSET_SMOOTH    = 6
CENTROID_SMOOTH = 10

BEAT_THRESHOLD_FACTOR = 1.6
BEAT_MIN_GAP          = 8    # frames (~370 ms) — suppresses double-triggers
TRANSIENT_FACTOR      = 2.2  # short/long RMS ratio to flag a transient


class AudioFeatures:
    __slots__ = [
        "rms", "smoothed_rms",
        "sub_bass", "bass", "mid", "high",
        "centroid",
        "beat", "transient", "beat_strength",
        "bpm",            # estimated tempo in BPM (smoothed, 60-200 range)
        "beat_phase",     # 0..1 position within current beat cycle
        "spectral_flux",  # half-wave rectified spectral flux (onset energy velocity)
        "spectral_rolloff",  # Hz below which 85% of energy lies
        "zcr",            # zero crossing rate [0,1] — roughness/noisiness
        "harmonic_ratio", # spectral stability [0,1] — 1=tonal, 0=percussive/noisy
        "dynamic_range",  # short/long RMS ratio — expressive vs compressed dynamics
        "bpm_delta",      # BPM rate-of-change (BPM/s): >0 speeding up, <0 slowing down
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0.0)
        self.beat = False
        self.transient = False
        self.bpm = 120.0
        self.beat_phase = 0.0
        self.spectral_rolloff = 4000.0
        self.harmonic_ratio = 0.5
        self.dynamic_range = 1.0
        self.bpm_delta = 0.0

    def to_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


class AudioIngest:
    def __init__(self):
        self._rate  = RATE
        self._chunk = CHUNK

        # FFT frequency axis
        self._freqs = np.fft.rfftfreq(CHUNK, d=1.0 / RATE)

        # Pre-compute boolean masks for each frequency band
        self._band_masks = {
            name: (self._freqs >= lo) & (self._freqs < hi)
            for name, (lo, hi) in BANDS.items()
        }

        # Hann window to reduce spectral leakage
        self._window_fn = np.hanning(CHUNK)

        # Smoothing buffers
        self._rms_buf      = deque(maxlen=RMS_SMOOTH)
        self._band_bufs    = {n: deque(maxlen=BAND_SMOOTH) for n in BANDS}
        self._onset_buf    = deque(maxlen=ONSET_SMOOTH)
        self._centroid_buf = deque(maxlen=CENTROID_SMOOTH)

        # Beat/onset detection state
        self._prev_spectrum     = np.zeros(CHUNK // 2 + 1)
        self._flux_history      = deque(maxlen=30)
        self._frames_since_beat = BEAT_MIN_GAP  # start ready

        # BPM estimation — inter-beat-interval history
        # Each beat stores the timestamp in seconds
        self._beat_times        = deque(maxlen=16)   # last 16 beat timestamps
        self._smoothed_bpm      = 120.0              # running smoothed BPM
        self._frame_time        = 0.0                # wall-clock seconds elapsed
        self._frame_dur         = CHUNK / RATE       # seconds per audio frame

        # Transient detection state (short vs long RMS window)
        self._rms_short = deque(maxlen=4)
        self._rms_long  = deque(maxlen=40)

        # New feature smoothing buffers
        self._flux_smooth_buf     = deque(maxlen=6)
        self._rolloff_buf         = deque(maxlen=10)
        self._zcr_buf             = deque(maxlen=8)
        self._harmonic_buf        = deque(maxlen=12)
        self._prev_spectrum_copy  = np.zeros(CHUNK // 2 + 1)  # kept before overwrite

        # BPM delta tracking (tempo acceleration)
        self._prev_smoothed_bpm   = 120.0
        self._bpm_delta_buf       = deque(maxlen=30)  # smooth over ~1.4s of frames

        # PyAudio
        self._p      = pyaudio.PyAudio()
        self._stream = None

        device_idx, device_rate, device_ch = self._find_loopback_device()
        if device_rate != self._rate:
            self._rate = device_rate
            self._freqs = np.fft.rfftfreq(CHUNK, d=1.0 / self._rate)
            self._band_masks = {
                name: (self._freqs >= lo) & (self._freqs < hi)
                for name, (lo, hi) in BANDS.items()
            }
            self._window_fn = np.hanning(CHUNK)
            self._frame_dur = CHUNK / self._rate

        try:
            self._stream = self._p.open(
                format=pyaudio.paInt16,
                channels=device_ch,
                rate=self._rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=self._chunk,
            )
            name = self._p.get_device_info_by_index(device_idx)["name"] if device_idx is not None else "default"
            print(f"[AudioIngest] Using device [{device_idx}] '{name}' @ {self._rate}Hz ch={device_ch}")
        except OSError as e:
            print(f"WARNING: Could not open audio input: {e}. Running without audio.")

    def _find_loopback_device(self):
        """
        Find the best system audio (loopback) device, never a microphone.
        """
        # --- 1. Look for 'Stereo Mix' or 'Loopback' devices ---
        LOOPBACK_KEYWORDS = ["stereo mix", "what u hear", "loopback", "wave out mix", "sum", "speakers wave", "streaming speakers", "virtual"]
        MIC_KEYWORDS      = ["microphone", "mic array", "webcam", "headset", "input mic", "streaming mic", "communications"]

        candidates = []
        for i in range(self._p.get_device_count()):
            try:
                d = self._p.get_device_info_by_index(i)
                if d["maxInputChannels"] < 1:
                    continue
                name_lc = d["name"].lower()
                # Prioritize things that match loopback but NOT mic
                if any(k in name_lc for k in LOOPBACK_KEYWORDS) and not any(k in name_lc for k in MIC_KEYWORDS):
                    candidates.append(i)
            except:
                continue

        # Try these candidates with multiple rates
        for idx in candidates:
            d = self._p.get_device_info_by_index(idx)
            # Try 44100 first as it's the most compatible with Stereo Mix usually
            for rate in [44100, 48000, 32000]:
                try:
                    ch = min(int(d["maxInputChannels"]), 2)
                    test = self._p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True, input_device_index=idx)
                    test.close()
                    print(f"[AudioIngest] SUCCESS: Captured loopback on [{idx}] '{d['name']}' @ {rate}Hz")
                    return idx, rate, ch
                except:
                    continue

        # --- 2. Last Resort: Search ALL non-microphone inputs ---
        print("[AudioIngest] No obvious loopback found. Searching all input devices...")
        for i in range(self._p.get_device_count()):
            try:
                d = self._p.get_device_info_by_index(i)
                if d["maxInputChannels"] < 1: continue
                name_lc = d["name"].lower()
                if not any(k in name_lc for k in MIC_KEYWORDS):
                    for rate in [44100, 48000]:
                        try:
                            ch = min(int(d["maxInputChannels"]), 2)
                            test = self._p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True, input_device_index=i)
                            test.close()
                            print(f"[AudioIngest] Found candidate: [{i}] '{d['name']}' @ {rate}Hz")
                            return i, rate, ch
                        except: continue
            except: continue

        # --- 3. Fatal fallback ---
        print("\n" + "!"*60)
        print(" [AUDIO CRITICAL] NO SYSTEM LOOPBACK DETECTED")
        print(" Engine is likely capturing your MICROPHONE.")
        print(" ")
        print(" TO FIX 'Stereo Mix' Host Errors on Windows:")
        print(" 1. Right-click Speaker Icon -> Sounds -> Recording")
        print(" 2. Right-click 'Stereo Mix' -> Properties -> Advanced")
        print(" 3. Set Default Format to '16 bit, 44100 Hz (CD Quality)'")
        print(" 4. Uncheck 'Allow applications to take exclusive control'")
        print(" 5. Restart this engine.")
        print("!"*60 + "\n")

        return None, 44100, 1

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_features(self) -> AudioFeatures:
        """Read one chunk and return a fully populated AudioFeatures object."""
        feat = AudioFeatures()
        if self._stream is None:
            return feat

        try:
            raw = self._stream.read(self._chunk, exception_on_overflow=False)
        except IOError:
            return feat

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # ── RMS ──────────────────────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self._rms_buf.append(rms)
        self._rms_short.append(rms)
        self._rms_long.append(rms)
        feat.rms = rms
        feat.smoothed_rms = float(np.mean(self._rms_buf))

        # ── Spectral analysis ─────────────────────────────────────────────────
        spectrum = np.abs(np.fft.rfft(samples * self._window_fn))

        # Band energies (mean magnitude per band, smoothed)
        for name, mask in self._band_masks.items():
            energy = float(np.mean(spectrum[mask])) if mask.any() else 0.0
            self._band_bufs[name].append(energy)
            setattr(feat, name, float(np.mean(self._band_bufs[name])))

        # Spectral centroid → brightness [0,1]
        total = float(np.sum(spectrum))
        if total > 1e-6:
            raw_c = float(np.sum(self._freqs * spectrum) / total)
        else:
            raw_c = 0.0
        norm_c = float(np.clip((raw_c - 20.0) / (8000.0 - 20.0), 0.0, 1.0))
        self._centroid_buf.append(norm_c)
        feat.centroid = float(np.mean(self._centroid_buf))

        # ── Zero crossing rate — roughness/noisiness ─────────────────────────
        zcr_raw = float(np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples)))
        self._zcr_buf.append(zcr_raw)
        feat.zcr = float(np.mean(self._zcr_buf))

        # ── Harmonic ratio — tonal stability vs percussive noise ──────────────
        # Pearson correlation of successive spectra: high = stable/tonal, low = noisy
        sp_norm = spectrum / (np.linalg.norm(spectrum) + 1e-10)
        prev_norm = self._prev_spectrum_copy / (np.linalg.norm(self._prev_spectrum_copy) + 1e-10)
        h_raw = float(np.dot(sp_norm, prev_norm))  # cosine similarity [0,1]
        self._harmonic_buf.append(max(0.0, h_raw))
        feat.harmonic_ratio = float(np.mean(self._harmonic_buf))

        # ── Spectral rolloff — frequency below which 85% of energy lies ──────
        cumsum = np.cumsum(spectrum)
        total_e = cumsum[-1]
        if total_e > 1e-6:
            rolloff_idx = int(np.searchsorted(cumsum, 0.85 * total_e))
            rolloff_idx = min(rolloff_idx, len(self._freqs) - 1)
            rolloff_hz = float(self._freqs[rolloff_idx])
        else:
            rolloff_hz = 0.0
        self._rolloff_buf.append(rolloff_hz)
        feat.spectral_rolloff = float(np.mean(self._rolloff_buf))

        # ── Beat/onset detection via half-wave rectified spectral flux ────────
        flux = float(np.sum(np.maximum(spectrum - self._prev_spectrum, 0.0)))
        self._prev_spectrum_copy = spectrum.copy()  # save before overwrite (for harmonic ratio)
        self._prev_spectrum = spectrum.copy()
        self._flux_history.append(flux)

        if len(self._flux_history) >= 10:
            threshold = float(np.median(self._flux_history)) * BEAT_THRESHOLD_FACTOR
        else:
            threshold = float('inf')

        self._onset_buf.append(flux)
        peak = max(float(np.max(self._flux_history)), 1.0) if self._flux_history else 1.0
        feat.beat_strength = float(np.clip(np.mean(self._onset_buf) / peak, 0.0, 1.0))

        # Expose smoothed spectral flux as a feature (normalized to [0,1])
        self._flux_smooth_buf.append(flux)
        feat.spectral_flux = float(np.clip(np.mean(self._flux_smooth_buf) / (peak + 1.0), 0.0, 1.0))

        self._frame_time += self._frame_dur
        self._frames_since_beat += 1
        if flux > threshold and self._frames_since_beat >= BEAT_MIN_GAP:
            feat.beat = True
            self._frames_since_beat = 0
            self._beat_times.append(self._frame_time)

        # ── BPM estimation from inter-beat intervals ──────────────────────────
        if len(self._beat_times) >= 2:
            intervals = np.diff(list(self._beat_times))
            # Reject outliers: keep intervals in [0.2s, 2.0s] (30-300 BPM)
            valid = intervals[(intervals > 0.2) & (intervals < 2.0)]
            if len(valid) >= 2:
                median_ibi = float(np.median(valid))
                raw_bpm = 60.0 / median_ibi
                # Clamp to musical range and smooth with slow EMA
                raw_bpm = max(60.0, min(200.0, raw_bpm))
                self._smoothed_bpm = self._smoothed_bpm * 0.92 + raw_bpm * 0.08

        feat.bpm = self._smoothed_bpm

        # ── BPM delta — tempo acceleration/deceleration ───────────────────────
        # Raw delta in BPM/s per frame, smoothed over ~1.4s
        raw_bpm_delta = (self._smoothed_bpm - self._prev_smoothed_bpm) / self._frame_dur
        self._bpm_delta_buf.append(raw_bpm_delta)
        feat.bpm_delta = float(np.mean(self._bpm_delta_buf))
        self._prev_smoothed_bpm = self._smoothed_bpm

        # Beat phase: position 0..1 within the current beat cycle
        if self._smoothed_bpm > 0:
            beat_dur = 60.0 / self._smoothed_bpm
            feat.beat_phase = (self._frame_time % beat_dur) / beat_dur

        # ── Transient: sudden RMS spike vs rolling baseline ───────────────────
        if len(self._rms_long) >= 10:
            short_mean = float(np.mean(self._rms_short))
            long_mean  = float(np.mean(self._rms_long))
            if long_mean > 10.0 and short_mean > long_mean * TRANSIENT_FACTOR:
                feat.transient = True

        # ── Dynamic range — expressiveness of performance ─────────────────────
        # Ratio of short (peak-following) to long (baseline) RMS.
        # >1 = expressive wide dynamics; ~1 = compressed/constant level
        if len(self._rms_long) >= 10 and float(np.mean(self._rms_long)) > 1.0:
            feat.dynamic_range = float(np.clip(
                float(np.mean(self._rms_short)) / float(np.mean(self._rms_long)),
                0.5, 5.0
            ))

        return feat

    # Legacy shim — main.py audio_thread still calls these
    def get_audio_intensity(self):
        return self.get_features().rms

    def get_smoothed_intensity(self):
        return float(np.mean(self._rms_buf)) if self._rms_buf else 0.0

    def stop(self):
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        self._p.terminate()


if __name__ == '__main__':
    ai = AudioIngest()
    print("Listening... Ctrl+C to stop.\n")
    try:
        while True:
            f = ai.get_features()
            beat_m = " *** BEAT ***"    if f.beat      else ""
            tran_m = " !! TRANSIENT !!" if f.transient else ""
            print(
                f"rms={f.smoothed_rms:7.1f}  "
                f"sub={f.sub_bass:6.1f}  bass={f.bass:6.1f}  "
                f"mid={f.mid:6.1f}  hi={f.high:6.1f}  "
                f"cent={f.centroid:.3f}  bstr={f.beat_strength:.3f}"
                f"{beat_m}{tran_m}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        ai.stop()
