import json
import random
from pydantic import BaseModel

# 1. Define the strict data schema for the VJ engine.
class VJStateSchema(BaseModel):
    macro_state: str
    primary_glyph: str
    color_palette: list[str]
    perlin_intensity: float
    video_prompt: str
    motion_intensity: float
    story_step: int  # 1-8 based on Dan Harmon's Story Circle
    narrative_beat: str # Short description of the current story moment

def create_vj_prompt(audio_telemetry: dict) -> str:
    """
    Creates a sophisticated prompt to give the LLM a creative persona.
    """
    style_guide = """
    -   **AUDIO-STORY SYNTHESIS:** The audio doesn't just drive motion; it drives the *fate* of the story.
        - High RMS/Energy: The story moves toward conflict, action, or transformation (Take, Search).
        - Low RMS/Calm: The story focuses on reflection, comfort, or discovery (You, Find).
        - Fast BPM (>140): Rapid story cuts, pursuit, or chaotic environments.
        - Slow BPM (<100): Slow-motion scenery, vast landscapes, or detailed interior shots.

    -   **VISUAL SUBJECTS:** Render **people, places, things, and landscapes**. Avoid "abstract textures".
    
    -   **STORY CIRCLE (Dan Harmon):** Ground each step in a physical reality tied to audio:
        1. YOU (Zone of Comfort) - Peaceful forest or cozy room. (Low energy audio).
        2. NEED (Internal Tension) - Unsettling details in a familiar place. (Subtle audio glitches).
        3. GO (Crossing Threshold) - Stepping into a massive desert or portal. (Audio buildup).
        4. SEARCH (Road of Trials) - Exploring dangerous canyons or neon cities. (Constant rhythm).
        5. FIND (The Goddess) - A hidden oasis or a glowing companion. (Melodic/Pure audio).
        6. TAKE (The Price) - A literal collapse, storm, or sacrifice. (The Audio Drop/High intensity).
        7. RETURN (Bring it back) - Flying home over mountains. (High energy resolution).
        8. CHANGE (Mastery) - A new skyline or a person with glowing eyes. (Steady, evolved audio).
    
    -   **video_prompt:** Describe cinematic scenes (e.g., "A weathered traveler climbing a sheer obsidian cliff as the bass pulses").
    -   **motion_intensity:** Link to the physical action described in the story.
    """

    # The JSON schema definition for the AI to follow.
    json_schema = VJStateSchema.model_json_schema()

    # The final prompt sent to the LLM.
    prompt = f"""You are VJ-LLAMA, a world-class AI video jockey.
Your goal is to synthesize audio telemetry into a visual story using the Dan Harmon Story Circle.

**DO NOT** output any text. Your response MUST be only the raw JSON object.

**YOUR STYLE GUIDE:**
{style_guide}

**CURRENT AUDIO TELEMETRY:**
{json.dumps(audio_telemetry, indent=2)}

**TASK:** Analyze the RMS (loudness) and BPM (rhythm). If energy is peaking, advance the story's intensity. If energy is low, deepen the story's atmosphere.

**JSON SCHEMA:**
{json.dumps(json_schema, indent=2)}
"""
    return prompt

def generate_prompt():
    """Generates a text prompt for the AI image generator."""
    prompts = [
        "a vibrant, psychedelic explosion of colors",
        "geometric patterns pulsing with energy",
        "a surreal landscape of melting clocks and floating islands",
        "an abstract representation of a deep bassline",
        "a glitchy, futuristic cityscape at night",
    ]
    return random.choice(prompts)
