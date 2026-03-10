use rand::seq::SliceRandom;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

pub const QUALITY_SUFFIX: &str = "photorealistic, hyper-detailed, cinematic color grading, 8k resolution, volumetric god rays, sharp focus, natural lighting, film grain, Arri Alexa color science, anamorphic lens, shallow depth of field, award-winning cinematography, realistic textures, HDR";

pub const NEGATIVE_PROMPT: &str = "blurry, low quality, pixelated, distorted, ugly, watermark, text, signature, flat, dull, noise, jpeg artifacts, cropped, duplicate, morbid, poorly drawn, bad anatomy, monochrome, desaturated, brown, grey, beige, washed out, muddy colors, dim, underexposed, cartoon, anime, illustration, painting, sketch, 3d render, CGI";

struct WordBank {
    subjects: Vec<String>,
    descriptors: Vec<String>,
    styles: Vec<String>,
}

pub struct Susa {
    banks: HashMap<String, WordBank>,
    usage: HashMap<String, f64>,
    intensity_history: VecDeque<f32>,
    last_fx_time: f64,
}

impl Susa {
    pub fn new() -> Self {
        let mut banks = HashMap::new();
        
        // Low intensity bank
        banks.insert("low".to_string(), WordBank {
            subjects: vec![
                "luminous aurora ribbons in 12 distinct shades of violet, seafoam, amber, and indigo drifting through a dark sky".to_string(),
                "a vast bioluminescent ocean glowing with 8 separate layers of mocha, teal, crimson, and neon gold".to_string(),
                "translucent crystal lattices refracting spectral light into a prismatic rainbow of 10 jewel tones".to_string(),
                "gossamer silk clouds of pearlescent mist in a spectrum of 7 soft pastel colors floating in zero gravity".to_string(),
                "a moonlit forest canopy dripping with bioluminescent dew in 9 kaleidoscopic hues from jade to magenta".to_string(),
            ],
            descriptors: vec![
                "serene and otherworldly with a hyper-chromatic palette of 12 vibrant colors".to_string(),
                "meditative and hypnotic, pulsing with a complex 8-color gradient".to_string(),
                "quietly radiant with a diverse spectrum of 7 jewel-toned treasures".to_string(),
            ],
            styles: vec![
                "photorealistic macro photography, cinematic bokeh, mocha and teal color grade".to_string(),
                "golden hour cinematography, natural film grain, warm volumetric light".to_string(),
            ],
        });

        // Medium intensity bank
        banks.insert("medium".to_string(), WordBank {
            subjects: vec![
                "a swirling fractal vortex of 12 electric colors including magenta, blue, lime, and orange".to_string(),
                "neon-lit geometric lattices in 9 contrasting colors like hot coral, cyan, and emerald floating in space".to_string(),
                "dynamic synaptic network of glowing nodes in a spectrum of 10 neon colors on a black void".to_string(),
            ],
            descriptors: vec![
                "vibrant and kinetically charged with a prismatic array of 12 distinct colors".to_string(),
                "electric and hypnotic, a kaleidoscopic explosion of 10 shifting colors".to_string(),
            ],
            styles: vec![
                "cyberpunk street photography, neon reflections on wet asphalt, Blade Runner palette".to_string(),
                "concert photography, dramatic stage lighting, smoke and laser haze".to_string(),
            ],
        });

        // High intensity bank
        banks.insert("high".to_string(), WordBank {
            subjects: vec![
                "a supernova collapsing into a blinding singularity of 12 white-hot magenta, gold, and azure shades".to_string(),
                "reality itself tearing open, exposing 10 layers of chromatic chaos in a rainbow of neon light".to_string(),
                "electric arcs of 8 separate colors including scarlet and violet erupting from a fractured core".to_string(),
            ],
            descriptors: vec![
                "violently explosive and overwhelming with a maximalist 12-color palette".to_string(),
                "chaotic beyond comprehension, 10 distinct colors at full saturation simultaneously".to_string(),
            ],
            styles: vec![
                "explosive pyrotechnics photography, high-speed capture, fireball colors".to_string(),
                "lightning storm photography, extreme contrast, electric atmosphere".to_string(),
            ],
        });

        Self {
            banks,
            usage: HashMap::new(),
            intensity_history: VecDeque::with_capacity(30),
            last_fx_time: 0.0,
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

    pub fn generate_prompt(&mut self, features: &crate::audio::AudioFeatures, narrative: Option<String>) -> String {
        self.intensity_history.push_back(features.smoothed_rms);
        if self.intensity_history.len() > 30 { self.intensity_history.pop_front(); }
        
        let avg_rms: f32 = self.intensity_history.iter().sum::<f32>() / self.intensity_history.len() as f32;
        let level = if avg_rms < 500.0 { "low" } else if avg_rms < 4000.0 { "medium" } else { "high" };
        
        let bank = self.banks.get(level).unwrap();
        
        let subject = if let Some(ref n) = narrative {
            n.clone()
        } else {
            self.weighted_choice(&bank.subjects)
        };
        
        let descriptor = self.weighted_choice(&bank.descriptors);
        let style = self.weighted_choice(&bank.styles);
        
        let prompt = format!("{}, {}, {}", subject, descriptor, style);
        
        if narrative.is_none() {
            self.mark_used(subject.clone());
        }
        self.mark_used(descriptor);
        self.mark_used(style);
        
        format!("{}, {}", prompt, QUALITY_SUFFIX)
    }
}
