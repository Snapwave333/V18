"""
Color Theory Mapper — maps psychological/emotional state to color vocabulary
for Stable Diffusion prompts.

Theoretical basis
-----------------
Color-Emotion Associations (Plutchik 1980, Hemphill 1996, Boyatzis & Varghese 1994):
    - Red:    excitement, passion, anger, urgency
    - Orange: enthusiasm, warmth, vitality, creativity
    - Yellow: happiness, brightness, optimism, attention
    - Green:  natural, growth, calm, balance, envy
    - Blue:   calm, sadness, trust, depth, melancholy
    - Purple: mystery, luxury, spirituality, melancholy, power
    - White:  purity, clarity, openness, emptiness
    - Black:  power, darkness, sophistication, fear, void
    - Gold:   triumph, richness, majesty, nostalgia

Itten's Color Theory (1961) — temperature and weight:
    - Warm colors (red/orange/yellow): advancing, active, exciting
    - Cool colors (blue/green/purple): receding, passive, calming
    - Saturated: high energy, emotional intensity
    - Desaturated/muted: low energy, introspection, depression

Synesthetic color-sound mappings (Cytowic 1993, Ward 2008):
    - Low bass frequencies → deep red, dark purple, black
    - Mid frequencies → warm amber, green, earthy brown
    - High frequencies → pale yellow, bright cyan, white
    - Dissonance → clashing complementaries, neon against dark
    - Consonance/harmony → analogous warm palettes, golden hour

Outputs
-------
Returns a dict with SD-ready color vocabulary strings:
    palette_prompt   — short comma-separated color description for SD
    lighting_prompt  — lighting mood for SD
    atmosphere_prompt — atmospheric/weather mood for SD
    color_adjectives — list of color/light descriptors
"""


# ── Color palette definitions ────────────────────────────────────────────────
# Each entry: (palette_prompt, lighting_prompt, atmosphere_prompt, adjectives)

_PALETTES = {
    # ── High valence, high arousal ────────────────────────────────────────────
    "euphoric": (
        "blazing gold and vivid crimson, electric cyan accents",
        "dramatic rim lighting, lens flares, saturated colors",
        "golden hour, crystal clear air, high contrast",
        ["radiant", "luminous", "blazing", "saturated", "electric"],
    ),
    "frenetic": (
        "intense electric violet and neon orange on a deep charcoal backdrop",
        "harsh cinematic lighting, high-speed motion blur, sharp lens flares",
        "vibrant and chaotic realistic scene, high-energy atmosphere",
        ["vibrant", "energetic", "electric", "sharp", "dynamic"],
    ),

    # ── High valence, low arousal ─────────────────────────────────────────────
    "serene": (
        "soft sky blue and warm ivory, pale lavender accents",
        "soft diffuse golden light, gentle shadows, dreamy bokeh",
        "misty morning, calm air, pastel haze",
        ["soft", "pastel", "airy", "luminous", "serene"],
    ),
    "wistful": (
        "dusty rose and faded gold, muted teal accents",
        "low warm sidelight, rich shadows, film grain",
        "late afternoon haze, nostalgic atmosphere, faded light",
        ["warm", "faded", "nostalgic", "tender", "amber"],
    ),

    # ── Low valence, high arousal ─────────────────────────────────────────────
    "urgent": (
        "deep crimson and burning orange, dark charcoal",
        "harsh directional light, deep shadows, high contrast",
        "storm-lit, oppressive sky, urgent tension",
        ["deep", "fierce", "saturated", "harsh", "burning"],
    ),
    "ominous": (
        "deep forest green and obsidian black, eerie emerald highlights",
        "cinematic low-key lighting, deep volumetric shadows, moody atmosphere",
        "overcast and heavy realistic sky, threatening storm clouds",
        ["dark", "moody", "emerald", "cinematic", "heavy"],
    ),

    # ── Low valence, low arousal ──────────────────────────────────────────────
    "melancholic": (
        "slate blue and muted charcoal, silver-grey highlights",
        "flat overcast light, desaturated tones, deep shadows",
        "overcast grey sky, heavy air, still and quiet",
        ["muted", "grey", "cool", "still", "desolate"],
    ),
    "desolate": (
        "near-black and deep navy, ice blue accents",
        "minimal cold light, near-total darkness, isolated highlight",
        "bleak, barren, freezing night, dead silence",
        ["bleak", "cold", "near-black", "void", "frozen"],
    ),

    # ── Fallback ──────────────────────────────────────────────────────────────
    "dynamic": (
        "rich amber and deep teal, warm neutral tones",
        "cinematic natural light, balanced exposure",
        "dynamic atmosphere, shifting light",
        ["rich", "cinematic", "warm", "deep", "natural"],
    ),
}


# ── Timbre-to-color overlay ───────────────────────────────────────────────────
# Additional color modifiers based on timbral character (synesthetic mapping)
_TIMBRE_OVERLAYS = {
    "warm":   "warm amber and mahogany undertones",
    "bright": "bright cyan and pale gold highlights",
    "dark":   "deep indigo and charcoal shadows",
    "harsh":  "overdriven neon, clashing complementary accents",
    "pure":   "clean white light, crystal clarity, minimal palette",
}

# ── Dominance modifier ────────────────────────────────────────────────────────
_DOMINANCE_MODIFIERS = {
    "high": "bold, high-contrast, powerful composition",
    "mid":  "balanced tonal range",
    "low":  "soft, diffuse, intimate scale",
}


class ColorTheoryMapper:
    """
    Maps psychological attributes (from MusicPsychologyMapper) to
    SD-ready color/lighting vocabulary strings.
    """

    def map(self, psychology: dict) -> dict:
        """
        Parameters
        ----------
        psychology : dict from MusicPsychologyMapper.map()

        Returns
        -------
        dict with: palette_prompt, lighting_prompt, atmosphere_prompt,
                   color_adjectives, full_color_context (combined SD string)
        """
        mood     = psychology.get("mood_label", "dynamic")
        timbre   = psychology.get("timbre_class", "warm")
        dominance = psychology.get("dominance", 0.5)

        # Look up palette (fall back to dynamic)
        palette, lighting, atmosphere, adjectives = _PALETTES.get(
            mood, _PALETTES["dynamic"]
        )

        # Add timbre overlay to palette
        timbre_overlay = _TIMBRE_OVERLAYS.get(timbre, "")
        if timbre_overlay:
            palette = f"{palette}, {timbre_overlay}"

        # Dominance modifier for composition/contrast
        if dominance > 0.65:
            dom_mod = _DOMINANCE_MODIFIERS["high"]
        elif dominance < 0.35:
            dom_mod = _DOMINANCE_MODIFIERS["low"]
        else:
            dom_mod = _DOMINANCE_MODIFIERS["mid"]

        full_color_context = f"{palette}, {lighting}, {atmosphere}, {dom_mod}"

        return {
            "palette_prompt":    palette,
            "lighting_prompt":   lighting,
            "atmosphere_prompt": atmosphere,
            "color_adjectives":  adjectives,
            "dominance_mod":     dom_mod,
            "full_color_context": full_color_context,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: ColorTheoryMapper | None = None


def get_mapper() -> ColorTheoryMapper:
    global _instance
    if _instance is None:
        _instance = ColorTheoryMapper()
    return _instance
