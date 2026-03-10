use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub frame_times_ms: VecDeque<f32>,
    pub audio_processing_times_ms: VecDeque<f32>,
    pub render_times_ms: VecDeque<f32>,
    pub texture_update_times_ms: VecDeque<f32>,
    pub current_fps: f32,
    pub average_frame_time_ms: f32,
    pub min_frame_time_ms: f32,
    pub max_frame_time_ms: f32,
    pub target_fps: u32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            frame_times_ms: VecDeque::with_capacity(120),
            audio_processing_times_ms: VecDeque::with_capacity(60),
            render_times_ms: VecDeque::with_capacity(120),
            texture_update_times_ms: VecDeque::with_capacity(60),
            current_fps: 0.0,
            average_frame_time_ms: 0.0,
            min_frame_time_ms: f32::MAX,
            max_frame_time_ms: 0.0,
            target_fps: 60,
        }
    }
}

pub struct PerformanceMonitor {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    last_frame_time: Instant,
}

impl PerformanceMonitor {
    pub fn new(target_fps: u32) -> Self {
        let mut metrics = PerformanceMetrics::default();
        metrics.target_fps = target_fps;
        
        Self {
            metrics: Arc::new(Mutex::new(metrics)),
            last_frame_time: Instant::now(),
        }
    }

    pub fn start_frame(&mut self) -> FrameTimer {
        FrameTimer {
            start_time: Instant::now(),
            metrics: self.metrics.clone(),
            phase: FramePhase::Frame,
        }
    }

    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    pub fn print_performance_report(&self) {
        let metrics = self.get_metrics();
        println!("=== Performance Report ===");
        println!("Target FPS: {}", metrics.target_fps);
        println!("Current FPS: {:.1}", metrics.current_fps);
        println!("Average Frame Time: {:.2}ms", metrics.average_frame_time_ms);
        println!("Min Frame Time: {:.2}ms", metrics.min_frame_time_ms);
        println!("Max Frame Time: {:.2}ms", metrics.max_frame_time_ms);
        println!("Frame Time 95th percentile: {:.2}ms", self.calculate_percentile(&metrics.frame_times_ms, 0.95));
        println!("========================");
    }

    fn calculate_percentile(&self, values: &VecDeque<f32>, percentile: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted: Vec<f32> = values.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((sorted.len() - 1) as f32 * percentile) as usize;
        sorted[index]
    }
}

pub struct FrameTimer {
    start_time: Instant,
    metrics: Arc<Mutex<PerformanceMetrics>>,
    phase: FramePhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FramePhase {
    Frame,
    AudioProcessing,
    Render,
    TextureUpdate,
}

impl FrameTimer {
    pub fn transition(&mut self, new_phase: FramePhase) {
        let elapsed = self.start_time.elapsed().as_secs_f32() * 1000.0;
        
        {
            let mut metrics = self.metrics.lock().unwrap();
            
            match self.phase {
                FramePhase::Frame => {
                    metrics.frame_times_ms.push_back(elapsed);
                    if metrics.frame_times_ms.len() > 120 {
                        metrics.frame_times_ms.pop_front();
                    }
                    
                    // Update FPS calculations
                    if !metrics.frame_times_ms.is_empty() {
                        metrics.average_frame_time_ms = metrics.frame_times_ms.iter().sum::<f32>() / metrics.frame_times_ms.len() as f32;
                        metrics.current_fps = 1000.0 / metrics.average_frame_time_ms;
                        metrics.min_frame_time_ms = metrics.min_frame_time_ms.min(elapsed);
                        metrics.max_frame_time_ms = metrics.max_frame_time_ms.max(elapsed);
                    }
                }
                FramePhase::AudioProcessing => {
                    metrics.audio_processing_times_ms.push_back(elapsed);
                    if metrics.audio_processing_times_ms.len() > 60 {
                        metrics.audio_processing_times_ms.pop_front();
                    }
                }
                FramePhase::Render => {
                    metrics.render_times_ms.push_back(elapsed);
                    if metrics.render_times_ms.len() > 120 {
                        metrics.render_times_ms.pop_front();
                    }
                }
                FramePhase::TextureUpdate => {
                    metrics.texture_update_times_ms.push_back(elapsed);
                    if metrics.texture_update_times_ms.len() > 60 {
                        metrics.texture_update_times_ms.pop_front();
                    }
                }
            }
        }
        
        self.start_time = Instant::now();
        self.phase = new_phase;
    }
}

impl Drop for FrameTimer {
    fn drop(&mut self) {
        if self.phase != FramePhase::Frame {
            self.transition(FramePhase::Frame);
        }
    }
}

// Performance optimization utilities
pub struct PerformanceOptimizer {
    frame_time_target_ms: f32,
    dynamic_quality_enabled: bool,
    quality_level: u8, // 0-10, higher is better quality
}

impl PerformanceOptimizer {
    pub fn new(target_fps: u32) -> Self {
        Self {
            frame_time_target_ms: 1000.0 / target_fps as f32,
            dynamic_quality_enabled: true,
            quality_level: 10,
        }
    }

    pub fn should_reduce_quality(&self, current_frame_time_ms: f32) -> bool {
        if !self.dynamic_quality_enabled {
            return false;
        }
        
        // Reduce quality if frame time exceeds target by more than 20%
        current_frame_time_ms > self.frame_time_target_ms * 1.2
    }

    pub fn should_increase_quality(&self, current_frame_time_ms: f32) -> bool {
        if !self.dynamic_quality_enabled {
            return false;
        }
        
        // Increase quality if frame time is 20% better than target
        current_frame_time_ms < self.frame_time_target_ms * 0.8 && self.quality_level < 10
    }

    pub fn adjust_quality(&mut self, current_frame_time_ms: f32) {
        if self.should_reduce_quality(current_frame_time_ms) && self.quality_level > 0 {
            self.quality_level -= 1;
            println!("Reducing quality to level {} due to performance", self.quality_level);
        } else if self.should_increase_quality(current_frame_time_ms) && self.quality_level < 10 {
            self.quality_level += 1;
            println!("Increasing quality to level {} due to spare performance", self.quality_level);
        }
    }

    pub fn get_quality_level(&self) -> u8 {
        self.quality_level
    }

    pub fn set_dynamic_quality(&mut self, enabled: bool) {
        self.dynamic_quality_enabled = enabled;
    }
}