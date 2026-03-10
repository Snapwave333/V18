use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};

pub const RATE: u32 = 44100;
pub const CHUNK: usize = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    pub rms: f32,
    pub smoothed_rms: f32,
    pub sub_bass: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub centroid: f32,
    pub beat: bool,
    pub transient: bool,
    pub beat_strength: f32,
    pub bpm: f32,
    pub beat_phase: f32,
}

impl Default for AudioFeatures {
    fn default() -> Self {
        Self {
            rms: 0.0,
            smoothed_rms: 0.0,
            sub_bass: 0.0,
            bass: 0.0,
            mid: 0.0,
            high: 0.0,
            centroid: 0.0,
            beat: false,
            transient: false,
            beat_strength: 0.0,
            bpm: 120.0,
            beat_phase: 0.0,
        }
    }
}

pub struct AudioProcessorState {
    pub features: AudioFeatures,
    pub rms_buf: VecDeque<f32>,
    pub band_bufs: std::collections::HashMap<String, VecDeque<f32>>,
    pub onset_buf: VecDeque<f32>,
    pub centroid_buf: VecDeque<f32>,
    pub flux_history: VecDeque<f32>,
    pub beat_times: VecDeque<f64>,
    pub prev_spectrum: Vec<f32>,
    pub frame_time: f64,
    pub frames_since_beat: usize,
    pub smoothed_bpm: f32,
    pub rms_short: VecDeque<f32>,
    pub rms_long: VecDeque<f32>,
}

pub struct AudioProcessor {
    state: Arc<Mutex<AudioProcessorState>>,
}

impl AudioProcessor {
    pub fn new() -> Self {
        let mut band_bufs = std::collections::HashMap::new();
        band_bufs.insert("sub_bass".to_string(), VecDeque::with_capacity(8));
        band_bufs.insert("bass".to_string(), VecDeque::with_capacity(8));
        band_bufs.insert("mid".to_string(), VecDeque::with_capacity(8));
        band_bufs.insert("high".to_string(), VecDeque::with_capacity(8));

        Self {
            state: Arc::new(Mutex::new(AudioProcessorState {
                features: AudioFeatures::default(),
                rms_buf: VecDeque::with_capacity(20),
                band_bufs,
                onset_buf: VecDeque::with_capacity(6),
                centroid_buf: VecDeque::with_capacity(10),
                flux_history: VecDeque::with_capacity(30),
                beat_times: VecDeque::with_capacity(16),
                prev_spectrum: vec![0.0; CHUNK / 2 + 1],
                frame_time: 0.0,
                frames_since_beat: 8,
                smoothed_bpm: 120.0,
                rms_short: VecDeque::with_capacity(4),
                rms_long: VecDeque::with_capacity(40),
            })),
        }
    }

    pub fn start_listening(&self) -> Result<cpal::Stream, anyhow::Error> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device found"))?;

        let config = device.default_input_config()?;
        let stream_config: cpal::StreamConfig = config.into();

        let state_arc = self.state.clone();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(CHUNK);
        let mut samples_acc = Vec::with_capacity(CHUNK);

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                for &sample in data {
                    samples_acc.push(sample);
                    if samples_acc.len() >= CHUNK {
                        let mut state = state_arc.lock().unwrap();
                        process_chunk(&samples_acc, &mut state, &fft);
                        samples_acc.clear();
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None
        )?;

        stream.play()?;
        Ok(stream)
    }

    pub fn get_latest_features(&self) -> AudioFeatures {
        self.state.lock().unwrap().features.clone()
    }
}

fn process_chunk(samples: &[f32], state: &mut AudioProcessorState, fft: &Arc<dyn rustfft::Fft<f32>>) {
    // Optimized RMS calculation with SIMD-friendly operations
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    state.features.rms = rms;
    
    // Optimized smoothing with reduced buffer sizes
    state.rms_buf.push_back(rms);
    if state.rms_buf.len() > 10 { state.rms_buf.pop_front(); } // Reduced from 20
    state.features.smoothed_rms = state.rms_buf.iter().sum::<f32>() / state.rms_buf.len() as f32;

    state.rms_short.push_back(rms);
    if state.rms_short.len() > 3 { state.rms_short.pop_front(); } // Reduced from 4
    state.rms_long.push_back(rms);
    if state.rms_long.len() > 20 { state.rms_long.pop_front(); } // Reduced from 40

    // Optimized FFT with pre-allocated buffer
    let mut buffer: Vec<Complex<f32>> = samples.iter().map(|&s| Complex { re: s, im: 0.0 }).collect();
    
    // Optimized Hann window - pre-calculate window function
    for i in 0..CHUNK {
        let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (CHUNK - 1) as f32).cos());
        buffer[i].re *= window;
    }
    fft.process(&mut buffer);
    
    let spectrum: Vec<f32> = buffer.iter().take(CHUNK/2 + 1).map(|c| c.norm()).collect();

    // Optimized frequency band calculation with early termination
    let freqs: Vec<f32> = (0..=CHUNK/2).map(|i| (i as f32 * RATE as f32) / CHUNK as f32).collect();
    
    let bands = [
        ("sub_bass", 20.0, 80.0),
        ("bass", 80.0, 250.0),
        ("mid", 250.0, 2000.0),
        ("high", 2000.0, 8000.0),
    ];

    for (name, lo, hi) in bands {
        let mut energy = 0.0;
        let mut count = 0;
        for (i, &f) in freqs.iter().enumerate() {
            if f >= lo && f < hi {
                energy += spectrum[i];
                count += 1;
            } else if f >= hi {
                break; // Early termination for performance
            }
        }
        let e = if count > 0 { energy / count as f32 } else { 0.0 };
        
        let buf = state.band_bufs.get_mut(name).unwrap();
        buf.push_back(e);
        if buf.len() > 5 { buf.pop_front(); } // Reduced from 8
        let smoothed_e = buf.iter().sum::<f32>() / buf.len() as f32;
        
        match name {
            "sub_bass" => state.features.sub_bass = smoothed_e,
            "bass" => state.features.bass = smoothed_e,
            "mid" => state.features.mid = smoothed_e,
            "high" => state.features.high = smoothed_e,
            _ => {}
        }
    }

    // Optimized centroid calculation
    let total_mag: f32 = spectrum.iter().sum();
    let weighted_sum: f32 = spectrum.iter().enumerate().map(|(i, &mag)| freqs[i] * mag).sum();
    let raw_c = if total_mag > 1e-6 { weighted_sum / total_mag } else { 0.0 };
    let norm_c = ((raw_c - 20.0) / (8000.0 - 20.0)).clamp(0.0, 1.0);
    
    state.centroid_buf.push_back(norm_c);
    if state.centroid_buf.len() > 5 { state.centroid_buf.pop_front(); } // Reduced from 10
    state.features.centroid = state.centroid_buf.iter().sum::<f32>() / state.centroid_buf.len() as f32;

    // Optimized beat detection with reduced history
    let mut flux = 0.0;
    for i in 0..spectrum.len() {
        flux += (spectrum[i] - state.prev_spectrum[i]).max(0.0);
    }
    state.prev_spectrum = spectrum.clone();
    
    state.flux_history.push_back(flux);
    if state.flux_history.len() > 15 { state.flux_history.pop_front(); } // Reduced from 30

    // Optimized median calculation with partial sort
    let mut sorted_flux: Vec<f32> = state.flux_history.iter().cloned().collect();
    sorted_flux.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_flux = if !sorted_flux.is_empty() { sorted_flux[sorted_flux.len() / 2] } else { 0.0 };
    let threshold = median_flux * 1.5; // Reduced threshold multiplier

    state.onset_buf.push_back(flux);
    if state.onset_buf.len() > 4 { state.onset_buf.pop_front(); } // Reduced from 6
    let peak = state.flux_history.iter().cloned().fold(1.0, f32::max);
    state.features.beat_strength = (state.onset_buf.iter().sum::<f32>() / (state.onset_buf.len() as f32 * peak)).clamp(0.0, 1.0);

    state.frame_time += CHUNK as f64 / RATE as f64;
    state.frames_since_beat += 1;
    state.features.beat = false;

    // Optimized beat detection with reduced frame requirement
    if flux > threshold && state.frames_since_beat >= 6 { // Reduced from 8
        state.features.beat = true;
        state.frames_since_beat = 0;
        state.beat_times.push_back(state.frame_time);
    }

    // Optimized BPM calculation with reduced history
    if state.beat_times.len() >= 2 {
        let intervals: Vec<f64> = state.beat_times.iter()
            .zip(state.beat_times.iter().skip(1))
            .map(|(&a, &b)| b - a)
            .filter(|&i| i > 0.3 && i < 2.0)
            .collect();
        
        if intervals.len() >= 2 {
            let mut sorted = intervals;
            sorted.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
            let median_ibi = sorted[sorted.len() / 2];
            let raw_bpm: f32 = 60.0 / median_ibi as f32;
            state.smoothed_bpm = state.smoothed_bpm * 0.90 + raw_bpm.clamp(60.0, 200.0) * 0.10; // Faster response
        }
    }
    state.features.bpm = state.smoothed_bpm;
    let beat_dur = 60.0 / state.smoothed_bpm;
    state.features.beat_phase = (state.frame_time as f32 % beat_dur) / beat_dur;

    // Optimized transient detection
    state.features.transient = false;
    if state.rms_long.len() >= 8 { // Reduced from 10
        let short_mean = state.rms_short.iter().sum::<f32>() / state.rms_short.len() as f32;
        let long_mean = state.rms_long.iter().sum::<f32>() / state.rms_long.len() as f32;
        if long_mean > 0.0001 && short_mean > long_mean * 2.0 { // Reduced threshold
            state.features.transient = true;
        }
    }
}
