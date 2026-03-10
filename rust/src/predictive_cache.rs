use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::VecDeque;
use std::time::{Instant, Duration};
use std::path::PathBuf;
use image::RgbaImage;
use serde::{Serialize, Deserialize};
use tokio::sync::mpsc;

// Serde helper for Instant
mod serde_millis {
    use serde::{self, Deserializer, Serializer, Deserialize};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let system_now = SystemTime::now();
        let instant_now = Instant::now();
        let duration_since_epoch = system_now.duration_since(UNIX_EPOCH).unwrap();
        let duration_since_instant = instant_now.duration_since(*instant);
        let system_time_at_instant = duration_since_epoch - duration_since_instant;
        serializer.serialize_u64(system_time_at_instant.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        let system_now = SystemTime::now();
        let instant_now = Instant::now();
        let duration_since_epoch = system_now.duration_since(UNIX_EPOCH).unwrap();
        let deserialized_duration = Duration::from_millis(millis);
        let duration_diff = if duration_since_epoch > deserialized_duration {
            duration_since_epoch - deserialized_duration
        } else {
            deserialized_duration - duration_since_epoch
        };
        Ok(instant_now - duration_diff)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedFrame {
    pub id: u64,
    pub prompt: String,
    #[serde(with = "serde_millis")]
    pub timestamp: Instant,
    pub audio_features_snapshot: AudioFeaturesSnapshot,
    pub journey_phase: String,
    pub spiritual_intensity: f32,
    pub estimated_beat_phase: f32,
    pub image_path: PathBuf,
    pub status: FrameStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FrameStatus {
    Predicting,
    Generating,
    Ready,
    Displayed,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeaturesSnapshot {
    pub rms: f32,
    pub smoothed_rms: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub centroid: f32,
    pub beat: bool,
    pub beat_strength: f32,
    pub bpm: f32,
    pub beat_phase: f32,
    pub transient: bool,
}

impl From<&crate::audio::AudioFeatures> for AudioFeaturesSnapshot {
    fn from(features: &crate::audio::AudioFeatures) -> Self {
        Self {
            rms: features.rms,
            smoothed_rms: features.smoothed_rms,
            bass: features.bass,
            mid: features.mid,
            high: features.high,
            centroid: features.centroid,
            beat: features.beat,
            beat_strength: features.beat_strength,
            bpm: features.bpm,
            beat_phase: features.beat_phase,
            transient: features.transient,
        }
    }
}

pub struct PredictiveFrameCache {
    cache: Arc<Mutex<VecDeque<PredictedFrame>>>,
    prediction_engine: Arc<PredictionEngine>,
    generation_tx: mpsc::Sender<GenerationRequest>,
    generation_rx: Arc<Mutex<mpsc::Receiver<GenerationRequest>>>,
    config: CacheConfig,
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub cache_size: usize,
    pub prediction_horizon_frames: usize,
    pub generation_timeout: Duration,
    pub frame_lifetime: Duration,
    pub prediction_interval: Duration,
    pub min_intensity_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_size: 10, // Keep 10 frames cached
            prediction_horizon_frames: 5, // Look 5 frames ahead
            generation_timeout: Duration::from_secs(30),
            frame_lifetime: Duration::from_secs(60),
            prediction_interval: Duration::from_millis(200), // Predict every 200ms
            min_intensity_threshold: 0.1,
        }
    }
}

#[derive(Debug)]
struct GenerationRequest {
    frame_id: u64,
    prompt: String,
    audio_snapshot: AudioFeaturesSnapshot,
    urgency: GenerationUrgency,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationUrgency {
    Low,    // For distant future frames
    Medium, // For near future frames
    High,   // For immediate next frame
    Critical, // Emergency generation
}

struct PredictionEngine {
    frame_counter: Arc<Mutex<u64>>,
    audio_history: Arc<Mutex<VecDeque<AudioFeaturesSnapshot>>>,
    pattern_recognizer: PatternRecognizer,
}

struct PatternRecognizer {
    beat_patterns: Vec<BeatPattern>,
    intensity_trends: Vec<IntensityTrend>,
    phase_transitions: Vec<PhaseTransition>,
}

#[derive(Debug, Clone)]
struct BeatPattern {
    bpm: f32,
    beat_strength: f32,
    phase_accuracy: f32,
    duration: Duration,
}

#[derive(Debug, Clone)]
struct IntensityTrend {
    start_intensity: f32,
    end_intensity: f32,
    duration: Duration,
    trend_type: TrendType,
}

#[derive(Debug, Clone)]
enum TrendType {
    Rising,
    Falling,
    Peaking,
    Valley,
    Stable,
}

#[derive(Debug, Clone)]
struct PhaseTransition {
    from_phase: String,
    to_phase: String,
    intensity_threshold: f32,
    bpm_trigger: Option<f32>,
    beat_pattern: Option<String>,
}

impl PredictiveFrameCache {
    pub fn new(config: CacheConfig) -> Self {
        let (tx, rx) = mpsc::channel(100);
        let prediction_engine = Arc::new(PredictionEngine::new());
        
        Self {
            cache: Arc::new(Mutex::new(VecDeque::with_capacity(config.cache_size))),
            prediction_engine,
            generation_tx: tx,
            generation_rx: Arc::new(Mutex::new(rx)),
            config,
        }
    }

    pub async fn start_prediction_loop(&self) {
        let cache = self.cache.clone();
        let engine = self.prediction_engine.clone();
        let config = self.config.clone();
        let tx = self.generation_tx.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.prediction_interval);
            
            loop {
                interval.tick().await;
                
                // Clean up expired frames
                Self::cleanup_expired_frames(&cache, config.frame_lifetime).await;
                
                // Generate predictions for future frames
                if let Ok(predictions) = engine.predict_future_frames(config.prediction_horizon_frames).await {
                    for prediction in predictions {
                        // Check if we need to generate this frame
                        if Self::should_generate_frame(&cache, &prediction).await {
                            if let Err(e) = tx.send(GenerationRequest {
                                frame_id: prediction.id,
                                prompt: prediction.prompt.clone(),
                                audio_snapshot: prediction.audio_features_snapshot.clone(),
                                urgency: Self::calculate_urgency(&prediction),
                            }).await {
                                eprintln!("Failed to send generation request: {}", e);
                            }
                            
                            // Add prediction to cache
                            Self::add_prediction_to_cache(&cache, prediction).await;
                        }
                    }
                }
            }
        });
    }

    pub async fn get_next_frame(&self) -> Option<PredictedFrame> {
        let mut cache = self.cache.lock().await;
        
        // Find the most appropriate ready frame
        if let Some(index) = cache.iter().position(|frame| frame.status == FrameStatus::Ready) {
            let mut frame = cache.remove(index).unwrap();
            frame.status = FrameStatus::Displayed;
            
            // Add back to cache for tracking
            cache.push_back(frame.clone());
            
            // Trigger generation of new predictions
            self.request_new_predictions().await;
            
            Some(frame)
        } else {
            // No ready frames available, trigger emergency generation
            self.trigger_emergency_generation().await;
            None
        }
    }

    pub async fn update_audio_features(&self, features: &crate::audio::AudioFeatures) {
        self.prediction_engine.update_current_features(features).await;
    }

    pub async fn get_cache_status(&self) -> CacheStatus {
        let cache = self.cache.lock().await;
        let ready_count = cache.iter().filter(|f| f.status == FrameStatus::Ready).count();
        let generating_count = cache.iter().filter(|f| f.status == FrameStatus::Generating).count();
        let predicting_count = cache.iter().filter(|f| f.status == FrameStatus::Predicting).count();
        
        CacheStatus {
            total_frames: cache.len(),
            ready_frames: ready_count,
            generating_frames: generating_count,
            predicting_frames: predicting_count,
            cache_hit_rate: self.calculate_hit_rate(),
            average_generation_time: self.get_average_generation_time(),
        }
    }

    async fn cleanup_expired_frames(cache: &Arc<Mutex<VecDeque<PredictedFrame>>>, lifetime: Duration) {
        let mut cache = cache.lock().await;
        let now = Instant::now();
        cache.retain(|frame| now.duration_since(frame.timestamp) < lifetime);
    }

    async fn should_generate_frame(cache: &Arc<Mutex<VecDeque<PredictedFrame>>>, prediction: &PredictedFrame) -> bool {
        let cache = cache.lock().await;
        !cache.iter().any(|frame| frame.id == prediction.id)
    }

    async fn add_prediction_to_cache(cache: &Arc<Mutex<VecDeque<PredictedFrame>>>, prediction: PredictedFrame) {
        let mut cache = cache.lock().await;
        if cache.len() >= cache.capacity() {
            cache.pop_front();
        }
        cache.push_back(prediction);
    }

    fn calculate_urgency(prediction: &PredictedFrame) -> GenerationUrgency {
        let time_until_needed = prediction.timestamp.elapsed().as_secs_f32();
        
        match time_until_needed {
            t if t < 0.5 => GenerationUrgency::Critical,
            t if t < 2.0 => GenerationUrgency::High,
            t if t < 5.0 => GenerationUrgency::Medium,
            _ => GenerationUrgency::Low,
        }
    }

    async fn request_new_predictions(&self) {
        // Implementation for requesting new predictions based on current state
        println!("Requesting new frame predictions...");
    }

    async fn trigger_emergency_generation(&self) {
        println!("TRIGGERING EMERGENCY FRAME GENERATION!");
        // Generate a simple frame immediately
    }

    fn calculate_hit_rate(&self) -> f32 {
        // Implementation for calculating cache hit rate
        0.85 // Placeholder
    }

    fn get_average_generation_time(&self) -> Duration {
        // Implementation for calculating average generation time
        Duration::from_secs(3) // Placeholder
    }
}

#[derive(Debug, Clone)]
pub struct CacheStatus {
    pub total_frames: usize,
    pub ready_frames: usize,
    pub generating_frames: usize,
    pub predicting_frames: usize,
    pub cache_hit_rate: f32,
    pub average_generation_time: Duration,
}

impl PredictionEngine {
    fn new() -> Self {
        Self {
            frame_counter: Arc::new(Mutex::new(0)),
            audio_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            pattern_recognizer: PatternRecognizer::new(),
        }
    }

    async fn predict_future_frames(&self, horizon: usize) -> Result<Vec<PredictedFrame>, String> {
        let mut predictions = Vec::new();
        let mut frame_id = self.frame_counter.lock().await;
        
        for i in 0..horizon {
            *frame_id += 1;
            let future_time = Instant::now() + Duration::from_millis(200 * (i + 1) as u64);
            
            // Predict future audio features
            let predicted_audio = self.predict_audio_features(i).await?;
            
            // Generate prompt based on predicted state
            let predicted_prompt = self.generate_predicted_prompt(&predicted_audio, i).await?;
            
            // Determine journey phase
            let predicted_phase = self.predict_journey_phase(&predicted_audio, i).await?;
            
            predictions.push(PredictedFrame {
                id: *frame_id,
                prompt: predicted_prompt,
                timestamp: future_time,
                audio_features_snapshot: predicted_audio.clone(),
                journey_phase: predicted_phase,
                spiritual_intensity: self.calculate_spiritual_intensity(&predicted_audio),
                estimated_beat_phase: self.estimate_beat_phase(&predicted_audio, i),
                image_path: PathBuf::from(format!("./predicted_frames/frame_{}.png", frame_id)),
                status: FrameStatus::Predicting,
            });
        }
        
        Ok(predictions)
    }

    async fn predict_audio_features(&self, frames_ahead: usize) -> Result<AudioFeaturesSnapshot, String> {
        let history = self.audio_history.lock().await;
        
        if history.is_empty() {
            return Ok(self.get_default_audio_snapshot());
        }
        
        // Simple prediction based on recent trends
        let recent = history.iter().rev().take(10).collect::<Vec<_>>();
        let avg_bpm: f32 = recent.iter().map(|h| h.bpm).sum::<f32>() / recent.len() as f32;
        let avg_intensity: f32 = recent.iter().map(|h| h.smoothed_rms).sum::<f32>() / recent.len() as f32;
        
        // Predict beat phase
        let last_beat_phase = recent.first().unwrap().beat_phase;
        let predicted_beat_phase = (last_beat_phase + 0.2 * frames_ahead as f32) % 1.0;
        
        Ok(AudioFeaturesSnapshot {
            rms: avg_intensity * (1.0 + 0.1 * (frames_ahead as f32).sin()),
            smoothed_rms: avg_intensity,
            bass: avg_intensity * 0.8,
            mid: avg_intensity * 0.6,
            high: avg_intensity * 0.4,
            centroid: 0.5 + 0.3 * (frames_ahead as f32 * 0.1).sin(),
            beat: predicted_beat_phase < 0.1 || predicted_beat_phase > 0.9,
            beat_strength: 0.7 + 0.3 * (predicted_beat_phase * std::f32::consts::PI * 2.0).sin(),
            bpm: avg_bpm,
            beat_phase: predicted_beat_phase,
            transient: frames_ahead == 2, // Predict transient at frame 2
        })
    }

    async fn generate_predicted_prompt(&self, audio: &AudioFeaturesSnapshot, frames_ahead: usize) -> Result<String, String> {
        // This would integrate with the enhanced_susa system
        // For now, return a placeholder that indicates prediction
        Ok(format!(
            "PREDICTED FRAME {}: Cosmic consciousness expanding with {}BPM rhythm, spiritual intensity {:.2}, beat phase {:.2}",
            frames_ahead, audio.bpm as i32, audio.beat_strength, audio.beat_phase
        ))
    }

    async fn predict_journey_phase(&self, audio: &AudioFeaturesSnapshot, _frames_ahead: usize) -> Result<String, String> {
        let phase = match audio.beat_strength {
            x if x < 0.3 => "awakening",
            x if x < 0.6 => "ascending",
            x if x < 0.8 => "transcending",
            _ => "unifying",
        };
        Ok(phase.to_string())
    }

    fn calculate_spiritual_intensity(&self, audio: &AudioFeaturesSnapshot) -> f32 {
        (audio.smoothed_rms / 1000.0).clamp(0.0, 1.0) * audio.beat_strength
    }

    fn estimate_beat_phase(&self, audio: &AudioFeaturesSnapshot, frames_ahead: usize) -> f32 {
        (audio.beat_phase + 0.2 * frames_ahead as f32) % 1.0
    }

    fn get_default_audio_snapshot(&self) -> AudioFeaturesSnapshot {
        AudioFeaturesSnapshot {
            rms: 0.5,
            smoothed_rms: 0.5,
            bass: 0.4,
            mid: 0.3,
            high: 0.2,
            centroid: 0.5,
            beat: false,
            beat_strength: 0.5,
            bpm: 120.0,
            beat_phase: 0.0,
            transient: false,
        }
    }

    async fn update_current_features(&self, features: &crate::audio::AudioFeatures) {
        let mut history = self.audio_history.lock().await;
        let snapshot = AudioFeaturesSnapshot::from(features);
        history.push_back(snapshot);
        if history.len() > 100 {
            history.pop_front();
        }
    }
}

impl PatternRecognizer {
    fn new() -> Self {
        Self {
            beat_patterns: vec![
                BeatPattern { bpm: 120.0, beat_strength: 0.8, phase_accuracy: 0.9, duration: Duration::from_secs(30) },
                BeatPattern { bpm: 140.0, beat_strength: 0.7, phase_accuracy: 0.85, duration: Duration::from_secs(45) },
                BeatPattern { bpm: 100.0, beat_strength: 0.9, phase_accuracy: 0.95, duration: Duration::from_secs(60) },
            ],
            intensity_trends: vec![],
            phase_transitions: vec![
                PhaseTransition { from_phase: "awakening".to_string(), to_phase: "ascending".to_string(), intensity_threshold: 0.2, bpm_trigger: None, beat_pattern: None },
                PhaseTransition { from_phase: "ascending".to_string(), to_phase: "transcending".to_string(), intensity_threshold: 0.5, bpm_trigger: Some(130.0), beat_pattern: Some("strong".to_string()) },
                PhaseTransition { from_phase: "transcending".to_string(), to_phase: "unifying".to_string(), intensity_threshold: 0.8, bpm_trigger: None, beat_pattern: Some("peak".to_string()) },
            ],
        }
    }
}