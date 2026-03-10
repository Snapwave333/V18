import json
import requests
import time
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
from prompt_generator import create_vj_prompt, VJStateSchema

# Configuration
GEMINI_API_KEY = "AIzaSyCUtMNor6v2SdbPNfwh5vAoFzD_3f9-6Z8"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
# Use Gemini 1.5 Flash for speed and reliability in VJ contexts
gemini_model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

# Global story state
_story_step = 1
_last_step_time = 0

def get_fallback_state() -> dict:
    """Returns the fail-safe VJ state."""
    return {
        "macro_state": "ACID_WASH",
        "primary_glyph": "#",
        "color_palette": ["#FF0000", "#000000"],
        "perlin_intensity": 1.0,
        "video_prompt": "a cinematic abstract background with pulsing lights",
        "motion_intensity": 0.5,
        "story_step": 1,
        "narrative_beat": "The journey begins in the zone of comfort."
    }

def fetch_ollama_fallback(prompt: str) -> dict:
    """Fallback mechanism using local Ollama."""
    print("[AI] Falling back to Ollama (Llama 3)...")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "temperature": 0.4
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=5.0)
        response.raise_for_status()
        state_str = response.json().get("response", "{}")
        state_dict = json.loads(state_str)
        VJStateSchema(**state_dict)
        return state_dict
    except Exception as e:
        print(f"[AI] Ollama fallback failed: {e}")
        return get_fallback_state()

def fetch_agent_state(audio_telemetry: dict) -> dict:
    """
    Queries Gemini as primary, falling back to Ollama on rate limits or errors.
    """
    global _story_step, _last_step_time
    
    # Progress story every 2 minutes (120s)
    if time.time() - _last_step_time > 120:
        _story_step = (_story_step % 8) + 1
        _last_step_time = time.time()
        print(f"[STORY] Advancing to Step {_story_step}")

    # Inject story context into telemetry
    audio_telemetry["current_story_step"] = _story_step

    prompt = create_vj_prompt(audio_telemetry)
    
    try:
        # 1. Attempt Gemini
        response = gemini_model.generate_content(prompt)
        
        # Parse result
        if response and response.text:
            state_dict = json.loads(response.text)
            # Validate against schema
            VJStateSchema(**state_dict)
            return state_dict
        else:
            raise ValueError("Empty response from Gemini")

    except Exception as e:
        # Check for rate limit (ResourceExhausted) or other errors
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            print("[AI] Gemini Rate Limit reached.")
        else:
            print(f"[AI] Gemini Error: {e}")
            
        # 2. Fallback to Ollama
        return fetch_ollama_fallback(prompt)
