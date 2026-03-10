mod audio;
mod susa;
mod storyteller;
mod agent;
mod performance;
mod predictive_cache;
mod enhanced_susa;

use audio::AudioProcessor;
use enhanced_susa::EnhancedSusa as Susa;
use storyteller::Storyteller;

use std::sync::{Arc, Mutex};
use std::time::Duration;
use actix_web::{web, App, HttpServer, HttpResponse, Responder};

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("1. Initializing Components...");
    let audio_processor = Arc::new(AudioProcessor::new());
    println!("   - Audio processor created.");
    let _audio_stream = match audio_processor.start_listening() {
        Ok(stream) => {
            println!("   - Audio stream started.");
            Some(stream)
        },
        Err(e) => {
            println!("   - WARNING: Failed to start audio stream: {}", e);
            println!("   - Continuing without audio reactivity.");
            None
        }
    };

    let _susa = Arc::new(Mutex::new(Susa::new()));
    let storyteller = Arc::new(Mutex::new(Storyteller::new()));
    let agent_bridge = Arc::new(agent::AgentBridge::new());

    println!("2. Starting Narrative & Agent Threads...");
    let st_clone = storyteller.clone();
    tokio::spawn(async move {
        println!("   - Storyteller thread starting...");
        Storyteller::run_loop(st_clone).await;
    });

    // AgentBridge runs on its own OS thread with its own tokio runtime.
    let ab_clone = agent_bridge.clone();
    let ap_clone = audio_processor.clone();
    std::thread::spawn(move || {
        println!("   - AgentBridge thread starting...");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(agent::AgentBridge::run_loop(ab_clone, ap_clone));
    });

    // 3. Telemetry writer (for Python AI worker)
    let audio_telemetry = audio_processor.clone();
    std::thread::spawn(move || {
        loop {
            let feat = audio_telemetry.get_latest_features();
            if let Ok(json) = serde_json::to_string(&feat) {
                let _ = std::fs::write("vj_features.json", json);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });

    // Resolve frames dir at startup
    let frames_dir = std::env::current_dir()
        .unwrap_or_default()
        .join("../python/temp_ipc/frames");
    println!("   - Frames dir: {:?}", frames_dir);

    // Predictive cache
    let cache_config = predictive_cache::CacheConfig::default();
    let frame_cache = Arc::new(predictive_cache::PredictiveFrameCache::new(cache_config));
    frame_cache.start_prediction_loop().await;

    let audio_processor_clone = audio_processor.clone();
    let cache_clone = frame_cache.clone();
    tokio::spawn(async move {
        loop {
            let feat = audio_processor_clone.get_latest_features();
            cache_clone.update_audio_features(&feat).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });

    // 4. HTTP Server (blocking — keeps process alive)
    println!("4. Starting HTTP server at http://127.0.0.1:8080");
    let ab_clone_web = agent_bridge.clone();
    let frames_dir_data = web::Data::new(frames_dir);
    HttpServer::new(move || {
        let abc = ab_clone_web.clone();
        App::new()
            .app_data(web::Data::new(abc))
            .app_data(frames_dir_data.clone())
            .route("/api/vj_state", web::get().to(get_vj_state))
            .route("/api/latest_frame", web::get().to(get_latest_frame))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await?;

    Ok(())
}

async fn get_vj_state(
    agent: web::Data<Arc<agent::AgentBridge>>,
) -> impl Responder {
    let state = agent.get_state();
    HttpResponse::Ok()
        .insert_header(("Access-Control-Allow-Origin", "*"))
        .json(state)
}

async fn get_latest_frame(frames_dir: web::Data<std::path::PathBuf>) -> impl Responder {
    let frame_path = frames_dir.join("latest_frame.png");

    if let Ok(data) = std::fs::read(&frame_path) {
        return HttpResponse::Ok()
            .insert_header(("Access-Control-Allow-Origin", "*"))
            .insert_header(("Content-Type", "image/png"))
            .insert_header(("Cache-Control", "no-cache"))
            .body(data);
    }
    HttpResponse::NotFound().finish()
}
