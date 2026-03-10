use image::RgbaImage;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use notify::{Watcher, RecursiveMode, Config};

pub struct AIFrame {
    pub image: RgbaImage,
    pub is_a: bool,
}

pub struct AIBridge {
    pub latest_frame: Arc<Mutex<Option<AIFrame>>>,
    watch_dir: PathBuf,
}

impl AIBridge {
    pub fn new(watch_dir: &str) -> Self {
        let watch_dir = PathBuf::from(watch_dir);
        if !watch_dir.exists() {
            std::fs::create_dir_all(&watch_dir).ok();
        }
        Self {
            latest_frame: Arc::new(Mutex::new(None)),
            watch_dir,
        }
    }

    pub fn start_watching(&self) {
        let latest_frame = self.latest_frame.clone();
        let watch_path = self.watch_dir.clone();

        std::thread::spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = notify::RecommendedWatcher::new(tx, Config::default()).unwrap();
            watcher.watch(&watch_path, RecursiveMode::NonRecursive).unwrap();

            let mut a_is_next = true;

            for res in rx {
                match res {
                    Ok(event) => {
                        for path in event.paths {
                            if path.extension().and_then(|s| s.to_str()) == Some("png") {
                                if let Ok(img) = image::open(&path) {
                                    let rgba = img.to_rgba8();
                                    let mut frame = latest_frame.lock().unwrap();
                                    *frame = Some(AIFrame {
                                        image: rgba,
                                        is_a: a_is_next,
                                    });
                                    println!("   [AI BRIDGE] Ingested new frame: {:?}", path.file_name().unwrap());
                                    a_is_next = !a_is_next;
                                }
                            }
                        }
                    }
                    Err(e) => println!("watch error: {:?}", e),
                }
            }
        });
    }

    pub fn get_latest_frame(&self) -> Option<AIFrame> {
        self.latest_frame.lock().unwrap().take()
    }
}
