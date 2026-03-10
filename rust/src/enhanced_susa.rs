use rand::seq::SliceRandom;
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// Enhanced quality suffix for mind-bending, psychedelic experiences
pub const QUALITY_SUFFIX: &str = "ultra-psychedelic, hyper-dimensional, consciousness-expanding, spiritual awakening, mind-bending fractals, transcendent light beings, interdimensional portals, kaleidoscopic mandalas, cosmic consciousness, synesthetic color explosions, ego dissolution, unity with the divine, sacred geometry, third eye activation, chakra energy vortexes, astral projection, quantum consciousness, infinite recursion, time dilation, reality distortion, luminous auras, ethereal mist, bioluminescent entities, morphing sacred symbols, prismatic light tunnels, nebula-like consciousness clouds, transcendent euphoria, divine revelation, cosmic oneness, 8K resolution, volumetric god rays, sharp focus, HDR, cinematic color grading";

pub const NEGATIVE_PROMPT: &str = "mundane, ordinary, realistic, boring, dull, flat, corporate, commercial, ugly, distorted, low quality, pixelated, watermark, text, signature, monochrome, desaturated, brown, grey, beige, washed out, muddy, dim, underexposed, cartoon, anime, illustration, painting, sketch, 3D render, CGI, plastic, artificial, fake, synthetic, lifeless, dead, dark, evil, scary, horror, violence, blood, gore, death, suffering, pain, fear, anxiety, depression, negativity";

struct PsychedelicWordBank {
    consciousness_states: Vec<String>,
    spiritual_realms: Vec<String>,
    fractal_dimensions: Vec<String>,
    color_phenomena: Vec<String>,
    reality_distortions: Vec<String>,
    divine_experiences: Vec<String>,
    cosmic_entities: Vec<String>,
    transcendent_moments: Vec<String>,
    sacred_geometries: Vec<String>,
    consciousness_expansions: Vec<String>,
}

pub struct EnhancedSusa {
    banks: HashMap<String, PsychedelicWordBank>,
    usage: HashMap<String, f64>,
    intensity_history: VecDeque<f32>,
    last_fx_time: f64,
    journey_phase: JourneyPhase,
    spiritual_intensity: f32,
    cosmic_alignment: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JourneyPhase {
    You,          // 1. Zone of comfort - Awakening
    Need,         // 2. Want something - Ascending
    Go,           // 3. Unfamiliar situation - Transcending
    Search,       // 4. Adapt to it - Dissolving
    Find,         // 5. Get what they wanted - Unifying
    Take,         // 6. Pay a heavy price - Integrating
    Return,       // 7. Return to familiar situation - Awakening (changed)
    Change,       // 8. Having changed - New cycle
}

impl EnhancedSusa {
    pub fn new() -> Self {
        let mut banks = HashMap::new();
        
        // 1. You - Zone of Comfort / Awakening
        banks.insert("you".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["gentle awareness in a familiar but luminous space".to_string()],
            spiritual_realms: vec!["a safe harbor for the soul, bathed in soft golden light".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 2. Need - Want Something / Ascending
        banks.insert("need".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["a yearning for something more, a pull toward the unknown".to_string()],
            spiritual_realms: vec!["a path unfolding, leading to a higher state of being".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 3. Go - Unfamiliar Situation / Transcending
        banks.insert("go".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["crossing the threshold into a reality beyond imagination".to_string()],
            spiritual_realms: vec!["a dimension of pure, chaotic, and beautiful creation".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 4. Search - Adapt to It / Dissolving
        banks.insert("search".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["navigating the new reality, letting go of old paradigms".to_string()],
            spiritual_realms: vec!["a fluid space where thoughts shape the environment".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 5. Find - Get What They Wanted / Unifying
        banks.insert("find".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["a moment of profound insight, a deep truth revealed".to_string()],
            spiritual_realms: vec!["the center of the mandala, the heart of cosmic consciousness".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 6. Take - Pay a Heavy Price / Integrating
        banks.insert("take".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["the journey's intensity subsides, leaving a profound change".to_string()],
            spiritual_realms: vec!["a quiet space for reflection, integrating the experience".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 7. Return - Return to Familiar / Awakening (Changed)
        banks.insert("return".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["returning to the familiar world, but seeing it with new eyes".to_string()],
            spiritual_realms: vec!["the sacred found in the ordinary, the divine in the mundane".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        // 8. Change - New Cycle
        banks.insert("change".to_string(), PsychedelicWordBank {
            consciousness_states: vec!["a new cycle begins, a new journey of transformation".to_string()],
            spiritual_realms: vec!["the universe invites you to another dance of creation".to_string()],
            fractal_dimensions: vec![],
            color_phenomena: vec![],
            reality_distortions: vec![],
            divine_experiences: vec![],
            cosmic_entities: vec![],
            transcendent_moments: vec![],
            sacred_geometries: vec![],
            consciousness_expansions: vec![],
        });

        Self {
            banks,
            usage: HashMap::new(),
            intensity_history: VecDeque::with_capacity(30),
            last_fx_time: 0.0,
            journey_phase: JourneyPhase::You,
            spiritual_intensity: 0.0,
            cosmic_alignment: 0.0,
        }
    }

    fn get_weight(&self, token: &str) -> f64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        if let Some(&last_used) = self.usage.get(token) {
            let elapsed = now - last_used;
            let half_life = 120.0;
            1.0 - (-elapsed / half_life * std::f64::consts::LN_2).exp()
        } else {
            1.0
        }
    }

    fn mark_used(&mut self, token: String) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        self.usage.insert(token, now);
    }

    fn weighted_choice(&self, tokens: &[String]) -> String {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = tokens.iter().map(|t| self.get_weight(t).max(0.05)).collect();
        tokens.choose_weighted(&mut rng, |item| {
            let idx = tokens.iter().position(|r| r == item).unwrap();
            weights[idx]
        }).unwrap().clone()
    }

    pub fn update_journey_phase(&mut self, audio_features: &crate::audio::AudioFeatures) {
        self.intensity_history.push_back(audio_features.smoothed_rms);
        if self.intensity_history.len() > 30 {
            self.intensity_history.pop_front();
        }
        
        let avg_intensity: f32 = self.intensity_history.iter().sum::<f32>() / self.intensity_history.len() as f32;
        let beat_strength = audio_features.beat_strength;
        let bpm = audio_features.bpm;
        
        // Calculate spiritual intensity based on audio features
        self.spiritual_intensity = (avg_intensity / 5000.0).clamp(0.0, 1.0) * beat_strength * (bpm / 180.0).clamp(0.5, 1.5);
        
        // Determine journey phase based on spiritual intensity and beat patterns
        self.journey_phase = match self.spiritual_intensity {
            x if x < 0.1 => JourneyPhase::You,      // Zone of comfort
            x if x < 0.2 => JourneyPhase::Need,     // Want something
            x if x < 0.4 => JourneyPhase::Go,       // Unfamiliar situation
            x if x < 0.6 => JourneyPhase::Search,   // Adapt to it
            x if x < 0.75 => JourneyPhase::Find,    // Get what they wanted
            x if x < 0.85 => JourneyPhase::Take,    // Pay a heavy price
            x if x < 0.95 => JourneyPhase::Return,  // Return to familiar
            _ => JourneyPhase::Change,             // Having changed
        };
        
        // Calculate cosmic alignment based on beat consistency and intensity
        if audio_features.beat && self.spiritual_intensity > 0.3 {
            self.cosmic_alignment = (self.cosmic_alignment * 0.9 + 0.1).min(1.0);
        } else {
            self.cosmic_alignment *= 0.95; // Gradual decay
        }
    }

    pub fn generate_enhanced_prompt(&mut self, features: &crate::audio::AudioFeatures, narrative: Option<String>) -> String {
        self.update_journey_phase(features);
        
        let phase_name = format!("{:?}", self.journey_phase).to_lowercase();
        let bank = self.banks.get(&phase_name).unwrap_or_else(|| self.banks.get("you").unwrap());
        
        // Select elements based on current spiritual intensity
        let mut prompt_elements = Vec::new();
        
        // Always include consciousness state
        prompt_elements.push(self.weighted_choice(&bank.consciousness_states));
        
        // Add spiritual realm based on intensity
        if self.spiritual_intensity > 0.2 {
            prompt_elements.push(self.weighted_choice(&bank.spiritual_realms));
        }
        
        // Add fractal dimensions for higher intensity
        if self.spiritual_intensity > 0.4 {
            prompt_elements.push(self.weighted_choice(&bank.fractal_dimensions));
        }
        
        // Add color phenomena based on beat
        if features.beat && self.spiritual_intensity > 0.3 {
            prompt_elements.push(self.weighted_choice(&bank.color_phenomena));
        }
        
        // Add reality distortions for peak experiences
        if self.spiritual_intensity > 0.6 {
            prompt_elements.push(self.weighted_choice(&bank.reality_distortions));
        }
        
        // Add divine experiences for transcendent moments
        if self.spiritual_intensity > 0.7 || self.cosmic_alignment > 0.8 {
            prompt_elements.push(self.weighted_choice(&bank.divine_experiences));
        }
        
        // Add cosmic entities for peak spiritual experiences
        if self.spiritual_intensity > 0.8 && features.beat {
            prompt_elements.push(self.weighted_choice(&bank.cosmic_entities));
        }
        
        // Build the final prompt
        let mut prompt = prompt_elements.join(", ");
        
        // Add special effects for cosmic alignment
        if self.cosmic_alignment > 0.7 {
            prompt.push_str(", cosmic alignment creating transcendent light bridges between dimensions");
        }
        
        if self.spiritual_intensity > 0.9 {
            prompt.push_str(", peak transcendent experience approaching unity consciousness");
        }
        
        // Add audio-reactive elements
        if features.beat {
            prompt.push_str(&format!(", pulsing with {}BPM cosmic rhythm", features.bpm as i32));
        }
        
        if features.transient {
            prompt.push_str(", reality shifting with consciousness expansion");
        }
        
        // Mark elements as used to avoid repetition
        for element in &prompt_elements {
            self.mark_used(element.clone());
        }
        
        format!("{}, {}", prompt, QUALITY_SUFFIX)
    }

    pub fn get_current_phase(&self) -> JourneyPhase {
        self.journey_phase.clone()
    }

    pub fn get_spiritual_intensity(&self) -> f32 {
        self.spiritual_intensity
    }

    pub fn get_cosmic_alignment(&self) -> f32 {
        self.cosmic_alignment
    }
}