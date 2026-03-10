use std::sync::{Arc, Mutex};
use crate::audio::AudioFeatures;
use serde::{Deserialize, Serialize};

fn default_macro_state() -> String { "ACID_WASH".to_string() }
fn default_glyph() -> String { "#".to_string() }
fn default_palette() -> Vec<String> { vec!["#FF0000".to_string(), "#00FFFF".to_string()] }
fn default_intensity() -> f32 { 0.5 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VJStateSchema {
    #[serde(default = "default_macro_state")]
    pub macro_state: String,
    #[serde(default = "default_glyph")]
    pub primary_glyph: String,
    #[serde(default = "default_palette")]
    pub color_palette: Vec<String>,
    #[serde(default = "default_intensity")]
    pub perlin_intensity: f32,
}

pub struct AgentBridge {
    pub current_state: Arc<Mutex<VJStateSchema>>,
}

impl AgentBridge {
    pub fn new() -> Self {
        Self {
            current_state: Arc::new(Mutex::new(get_fallback_state())),
        }
    }

    pub fn get_state(&self) -> VJStateSchema {
        self.current_state.lock().unwrap().clone()
    }

    pub async fn run_loop(bridge: Arc<Self>, audio: Arc<crate::audio::AudioProcessor>) {
        loop {
            let feat = audio.get_latest_features();
            // Convert features to a simple JSON telemetry object
            let telemetry = serde_json::json!({
                "rms": feat.smoothed_rms,
                "bpm": feat.bpm,
                "beat": feat.beat,
                "centroid": feat.centroid,
            });

            if let Some(new_state) = fetch_agent_state(&telemetry).await {
                let mut state = bridge.current_state.lock().unwrap();
                *state = new_state;
            }

            tokio::time::sleep(tokio::time::Duration::from_secs(4)).await;
        }
    }
}

pub fn get_fallback_state() -> VJStateSchema {
    VJStateSchema {
        macro_state: "ACID_WASH".to_string(),
        primary_glyph: "#".to_string(),
        color_palette: vec![
            "#FF0000".to_string(), "#FF7F00".to_string(), "#FFFF00".to_string(), 
            "#00FF00".to_string(), "#0000FF".to_string(), "#4B0082".to_string(), 
            "#9400D3".to_string(), "#FF1493".to_string(), "#00FFFF".to_string(),
            "#FF00FF".to_string()
        ],
        perlin_intensity: 1.0,
    }
}

pub async fn fetch_agent_state(audio_telemetry: &serde_json::Value) -> Option<VJStateSchema> {
    let client = reqwest::Client::new();
    let prompt = create_vj_prompt(audio_telemetry);

    let payload = serde_json::json!({
        "model": "qwen2.5:7b-instruct-q5_K_M",
        "prompt": prompt,
        "format": "json",
        "stream": false,
        "temperature": 0.4
    });

    println!("[AGENT] Sending request to Ollama...");
    match client.post("http://localhost:11434/api/generate")
        .json(&payload)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await {
            Ok(resp) => {
                println!("[AGENT] Got HTTP response: {}", resp.status());
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let state_str = data.get("response").and_then(|v| v.as_str()).unwrap_or("");
                        println!("[AGENT] response field: {}", &state_str[..state_str.len().min(120)]);
                        match serde_json::from_str::<VJStateSchema>(state_str) {
                            Ok(state) => { println!("[AGENT] Parsed OK: glyph={}", state.primary_glyph); return Some(state); }
                            Err(e) => println!("[AGENT] Parse error: {e}"),
                        }
                    }
                    Err(e) => println!("[AGENT] JSON decode error: {e}"),
                }
            }
            Err(e) => println!("[AGENT] Request error: {e}"),
        }

    None
}

fn create_vj_prompt(audio_telemetry: &serde_json::Value) -> String {
    let audio_str = serde_json::to_string(audio_telemetry).unwrap();
    let example = r##"{"macro_state":"PULSE","primary_glyph":"@","color_palette":["#FF0000","#00FF00","#0000FF","#FF00FF"],"perlin_intensity":0.7}"##;
    format!(
        "VJ visualizer state. Audio: {audio_str}. Output ONLY this JSON (no other text):\n{example}\nPick macro_state from [ACID_WASH,PULSE,DRIFT,CHAOS], primary_glyph from one of [@#&*+~.|], color_palette with 4-8 vibrant hex colors matching the energy, perlin_intensity 0.0-1.0."
    )
}
