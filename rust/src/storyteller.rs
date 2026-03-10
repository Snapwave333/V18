use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

pub const BEAT_INTERVAL: u64 = 45;
pub const OLLAMA_MODEL: &str = "llama3.2";

#[derive(Debug, Clone)]
pub struct StoryBeat {
    pub name: String,
    pub energy: String,
    pub archetype: String,
}

pub struct Storyteller {
    beat_index: usize,
    context: String,
    beat_name: String,
    audio_energy: String,
    beats: Vec<StoryBeat>,
}

impl Storyteller {
    pub fn new() -> Self {
        let beats = vec![
            StoryBeat {
                name: "YOU — Zone of Comfort".to_string(),
                energy: "calm".to_string(),
                archetype: "a still and comfortable sanctuary, soft golden light, familiar warmth, a figure at rest in a luminous haven".to_string(),
            },
            StoryBeat {
                name: "NEED — Desire / Problem".to_string(),
                energy: "yearning".to_string(),
                archetype: "a restless yearning, a glowing horizon calling from afar, something beautiful just out of reach, longing rendered in light".to_string(),
            },
            StoryBeat {
                name: "GO — Enter Unknown".to_string(),
                energy: "threshold".to_string(),
                archetype: "a threshold being crossed, a portal of light opening into darkness, the first step into the unknown, a world transforming around a traveler".to_string(),
            },
            StoryBeat {
                name: "SEARCH — Adapt & Explore".to_string(),
                energy: "searching".to_string(),
                archetype: "an explorer mapping a strange and beautiful world, labyrinthine pathways of glowing structures, discovery around every corner, wonder and danger intertwined".to_string(),
            },
            StoryBeat {
                name: "FIND — Finding What's Needed".to_string(),
                energy: "revelation".to_string(),
                archetype: "a moment of revelation and discovery, blinding light breaking through darkness, the answer revealed in radiant clarity, a cosmic prize grasped at last".to_string(),
            },
            StoryBeat {
                name: "TAKE — Pay the Price".to_string(),
                energy: "sacrifice".to_string(),
                archetype: "the cost of transformation, a shattering and painful price, beauty and destruction intertwined, something precious dissolving to make way for growth".to_string(),
            },
            StoryBeat {
                name: "RETURN — Come Back Changed".to_string(),
                energy: "return".to_string(),
                archetype: "a journey home through transformed landscapes, familiar places seen with new eyes, the old world now strange and wonderful, a figure carrying hard-won wisdom back through the threshold".to_string(),
            },
            StoryBeat {
                name: "CHANGE — Transformed Forever".to_string(),
                energy: "transcendence".to_string(),
                archetype: "total transformation and transcendence, a being reborn from their journey, the comfort zone remade into something richer and more complex, integration of light and shadow into a new whole".to_string(),
            },
        ];

        Self {
            beat_index: 0,
            context: beats[0].archetype.clone(),
            beat_name: beats[0].name.clone(),
            audio_energy: "calm".to_string(),
            beats,
        }
    }

    pub fn set_audio_energy(&mut self, level: &str, trend: &str) {
        self.audio_energy = match (level, trend) {
            ("low", "sustained") => "calm and meditative",
            ("low", "rising") => "gently building",
            ("low", "falling") => "softly dissolving",
            ("medium", "sustained") => "energetic and flowing",
            ("medium", "rising") => "intensifying",
            ("medium", "falling") => "winding down",
            ("high", "sustained") => "explosive and overwhelming",
            ("high", "rising") => "approaching climax",
            ("high", "falling") => "cathartic release",
            _ => "dynamic",
        }.to_string();
    }

    pub fn get_context(&self) -> String {
        self.context.clone()
    }

    pub fn get_beat_name(&self) -> String {
        self.beat_name.clone()
    }

    pub async fn run_loop(storyteller: Arc<Mutex<Self>>) {
        let client = reqwest::Client::new();
        loop {
            let (beat, energy) = {
                let s = storyteller.lock().unwrap();
                (s.beats[s.beat_index].clone(), s.audio_energy.clone())
            };

            let visual = fetch_visual_description(&client, &beat, &energy).await;

            {
                let mut s = storyteller.lock().unwrap();
                s.context = visual;
                s.beat_name = beat.name.clone();
                s.beat_index = (s.beat_index + 1) % s.beats.len();
            }

            sleep(Duration::from_secs(BEAT_INTERVAL)).await;
        }
    }
}

async fn fetch_visual_description(client: &reqwest::Client, beat: &StoryBeat, energy: &str) -> String {
    let prompt = format!(
        "Describe a stunning hyper-chromatic visual scene using a palette of exactly 10 distinct vibrant colors that captures '{}' with {} energy. Base it on this archetype: {}. Be poetic and specific. 20 words max. Output ONLY the description.",
        beat.name, energy, beat.archetype
    );

    let payload = serde_json::json!({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.9,
            "num_predict": 50
        }
    });

    match client.post("http://localhost:11434/api/generate")
        .json(&payload)
        .send()
        .await {
            Ok(resp) => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if let Some(text) = data.get("response").and_then(|v| v.as_str()) {
                        return text.trim().trim_matches('"').to_string();
                    }
                }
            }
            Err(_) => {}
        }

    beat.archetype.clone()
}
