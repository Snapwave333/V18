#version 330 core

uniform sampler2D u_texture_a;
uniform sampler2D u_texture_b;
uniform sampler2D u_ascii_atlas;  // 160x16 RGBA glyph strip
uniform float u_time;
uniform float u_crossfade;
uniform vec2  u_px;
uniform vec2  u_resolution;       // screen size in pixels
// Audio reactivity
uniform float u_rms;
uniform float u_bass;
uniform float u_high;
uniform float u_beat;
uniform float u_centroid;
// ASCII overlay
uniform float u_ascii_blend;      // 0=normal, 1=full ascii
uniform float u_cell_size;        // character cell size in pixels
uniform float u_beat_phase;       // 0..1 position within beat cycle
uniform float u_fx_mode;          // 0=none,1=mirror,2=quad,3=kaleido,4=edge
uniform float u_fx_blend;         // 0..1 strength of the dramatic FX
uniform float u_audio_presence;   // 0=silence→black, 1=full visuals
// AI-controllable parameters
uniform float u_ai_hue_shift;     // extra hue rotation (radians-ish)
uniform float u_ai_sat_boost;     // saturation modifier
uniform float u_ai_glow;          // glow intensity multiplier
uniform float u_ai_edge_width;    // edge detection kernel width
uniform float u_ai_edge_glow;     // edge neon glow intensity
uniform float u_splat_blend;      // 0=normal, 1=full gaussian splat

// Color Palette Support
uniform vec3  u_palette[12];
uniform int   u_palette_size;
uniform float u_palette_blend;    // 0=original, 1=mapped to palette

in  vec2 v_uv;
out vec4 f_color;

#define PI  3.14159265359
#define TAU 6.28318530718
#define ATLAS_COLS 16.0

// -- Hue rotation --
vec3 hueRotate(vec3 col, float angle) {
    float c = cos(angle), s = sin(angle);
    mat3 m = mat3(
        0.299+0.701*c-0.168*s, 0.587-0.587*c+0.330*s, 0.114-0.114*c-0.497*s,
        0.299-0.299*c+0.328*s, 0.587+0.413*c+0.035*s, 0.114-0.114*c-0.292*s,
        0.299-0.300*c-1.250*s, 0.587-0.588*c-1.050*s, 0.114+0.886*c+1.050*s
    );
    return clamp(m * col, 0.0, 1.0);
}

// ── Shared Visual Logic ───────────────────────────────────────────────────

// 1. Compute the raw image color from the warped UVs (Source Content)
//    Includes mixing A/B, Hue Shift, and Saturation.
vec3 computeRawColor(vec2 warpedUV, float cf) {
    vec3 colA = texture(u_texture_a, warpedUV).rgb;
    vec3 colB = texture(u_texture_b, warpedUV).rgb;
    vec3 col  = mix(colA, colB, cf);
    
    // Hue
    col = hueRotate(col, u_ai_hue_shift * 0.5);
    
    // Saturation
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float s = clamp(1.0 + u_rms * 0.15 + u_beat * 0.1 + u_ai_sat_boost, 0.8, 1.3);
    col = mix(vec3(l), col, s);
    
    return col;
}

// 2. Apply Post-Process Effects (Glow, Vignette, FX, Palette)
//    Requires the 'screenUV' (where we are on screen) to apply spatial effects.
//    Also applies Glow which samples neighbors of the warped source.
vec3 applyPostEffects(vec3 base, vec2 warpedUV, vec2 screenUV) {
    // -- Glow --
    float brightness = dot(base, vec3(0.333));
    float glowMask   = smoothstep(0.6, 0.95, brightness);
    vec2 blurPx      = u_px * (1.0 + u_bass * 5.0 + u_high * 2.5);
    
    // Sample neighbors (using warped UVs to stay attached to content)
    vec3 glow = (
        texture(u_texture_b, warpedUV + vec2( blurPx.x,  0.0)).rgb +
        texture(u_texture_b, warpedUV + vec2(-blurPx.x,  0.0)).rgb +
        texture(u_texture_b, warpedUV + vec2( 0.0,  blurPx.y)).rgb +
        texture(u_texture_b, warpedUV + vec2( 0.0, -blurPx.y)).rgb
    ) * 0.25;
    
    base += base * glowMask * (0.1 + u_bass * 0.2) * (1.0 + u_ai_glow);

    // -- Vignette --
    vec2 centred = screenUV - 0.5;
    float vig = clamp(pow(1.0 - dot(centred * 1.3, centred * 1.3), 0.5), 0.0, 1.0);
    base *= vig;

    // -- FX Modes (Mirror/Quad) --
    // These overlay raw texture B usually? 
    // Ideally they should mirror the *current visual*.
    // Converting them to act on 'base' implies mapping logic. 
    // The original code re-sampled u_texture_b. We will preserve that behavior.
    float fxB = smoothstep(0.0, 1.0, u_fx_blend);
    if (u_fx_mode == 1.0) { // Mirror
        vec2 mUV = vec2(abs(screenUV.x - 0.5) + 0.5, screenUV.y);
        vec3 mC = texture(u_texture_b, mUV).rgb; // Raw sample
        base = mix(base, mC, fxB);
    } else if (u_fx_mode == 2.0) { // Quad
        vec2 qUV = vec2(abs(screenUV.x - 0.5), abs(screenUV.y - 0.5)) * 2.0;
        vec3 qC = texture(u_texture_b, qUV).rgb; // Raw sample
        base = mix(base, qC, fxB);
    }

    // -- Palette --
    if (u_palette_size > 0 && u_palette_blend > 0.001) {
        float l = dot(base, vec3(0.299, 0.587, 0.114));
        float pIdx = l * float(u_palette_size - 1);
        int i0 = int(floor(pIdx));
        int i1 = int(ceil(pIdx));
        vec3 pCol = mix(u_palette[i0], u_palette[i1], fract(pIdx));
        base = mix(base, pCol, u_palette_blend);
    }
    
    return base;
}

// ── ASCII Overlay ─────────────────────────────────────────────────────────

vec4 asciiLayer(vec2 fragCoord, vec2 warpedUV, vec2 screenUV, float cf, float cellSizePx) {
    // 1. Quantize to Cell Grid
    // We want the visual properties of the CELL CENTER.
    
    // Map current pixel's warped UV to the cell grid space
    vec2 gridUV = warpedUV * u_resolution / cellSizePx;
    vec2 cellCenterGrid = floor(gridUV) + 0.5;
    vec2 cellCenterWarpedUV = cellCenterGrid * cellSizePx / u_resolution;
    
    // Map current pixel's Screen UV to cell grid space (for vignette/FX)
    vec2 screenGridUV = screenUV * u_resolution / cellSizePx;
    vec2 cellCenterScreenUV = (floor(screenGridUV) + 0.5) * cellSizePx / u_resolution; // Fixed: use screenGridUV
    
    // 2. Compute "GLSL Layer" color at the cell center
    vec3 cellColor = computeRawColor(cellCenterWarpedUV, cf);
    cellColor = applyPostEffects(cellColor, cellCenterWarpedUV, cellCenterScreenUV);

    // 3. Determine Glyph
    float lum = dot(cellColor, vec3(0.299, 0.587, 0.114));
    float charIdx = floor(lum * 15.0 + 0.5);
    
    // 4. Sample Atlas
    vec2 localUV = fract(gridUV); // Fraction of the grid cell
    float atlasU = (charIdx + localUV.x) / ATLAS_COLS;
    float atlasV = localUV.y;
    float glyphAlpha = texture(u_ascii_atlas, vec2(atlasU, atlasV)).r;
    
    // 5. Lighting / Edge (Optional, but adds depth)
    // We'll trust the color luminance for now.
    
    float sAlpha = smoothstep(0.4, 0.6, glyphAlpha);
    
    // 6. Return Colored Glyph
    // We behave like a "colored text" layer.
    return vec4(cellColor, sAlpha); // Alpha is the shape, Color is the GLSL color
}


void main() {
    float t  = u_time;
    
    // ── Motion & Coordinates ──────────────────────────────────────────────
    float zoom = 1.0 + 0.05 * sin(t * 0.2); // +/- 5% breathing zoom
    vec2 pan   = vec2(sin(t * 0.1), cos(t * 0.13)) * 0.02; // Slow orbital pan
    vec2 uv = (v_uv - 0.5) * zoom + 0.5 + pan;

    float drift = u_centroid * 0.005 + 0.002;
    vec2 w2 = uv + vec2(
        sin(t * 0.08 + uv.y * 3.14) * drift,
        cos(t * 0.07 + uv.x * 3.14) * drift
    );

    // ── Optical Flow Warp ──────────────────────────────────────────────
    float cf = smoothstep(0.0, 1.0, clamp(u_crossfade, 0.0, 1.0));
    
    float lumA = dot(texture(u_texture_a, w2).rgb, vec3(0.299, 0.587, 0.114));
    float lumB = dot(texture(u_texture_b, w2).rgb, vec3(0.299, 0.587, 0.114));
    
    vec2 flowDir = vec2(dFdx(lumB) - dFdx(lumA), dFdy(lumB) - dFdy(lumA));
    float flowStr = (0.05 + u_bass * 0.1) * sin(cf * PI); 
    
    vec2 uvA = w2 + flowDir * flowStr;
    vec2 uvB = w2 - flowDir * flowStr;
    vec2 warpedUV = mix(uvA, uvB, cf);

    // ── Compute Base GLSL Layer ────────────────────────────────────────
    // This is the "Truth" of what the shader looks like.
    vec3 base = computeRawColor(warpedUV, cf);
    base = applyPostEffects(base, warpedUV, uv);

    // ── ASCII Overlay ──────────────────────────────────────────────────
    // Transparent Mix Mode:
    // We want the GLSL layer (base) to be the background.
    // We want the ASCII glyphs to be drawn on top.
    // The ASCII glyphs should match the color of the GLSL layer at that point (Mapping).
    // To make them visible against the *same* background, we bump brightness or alpha.
    
    if (u_ascii_blend > 0.001) {
        vec2 fragCoord = v_uv * u_resolution; // Use raw v_uv for grid alignment? No, use motion uv?
        // Actually, if we use 'uv' (zoomed), the grid zooms too.
        // If we want fixed grid size on screen, we use v_uv.
        // But then the text slides over the content.
        // Usually ASCII art implies the grid is fixed to the screen (terminal).
        // So we use 'v_uv' for grid, but we map content from 'warpedUV'.
        // YES.
        
        float asciiBlend = smoothstep(0.0, 1.0, u_ascii_blend);
        vec4 ascii = asciiLayer(fragCoord, warpedUV, uv, cf, u_cell_size);
        
        // Composition:
        // base = background.
        // ascii.rgb = color of glyph (from glsl layer).
        // ascii.a = shape of glyph.
        
        // Boost ascii brightness to pop against self
        vec3 asciiFg = ascii.rgb * 1.5; 
        
        // Mix: Background * (1-a) + Foreground * a  <-- Standard Alpha
        // But we want transparency, so we keep background? 
        // mix(base, asciiFg, ascii.a) replaces the background inside the glyph.
        base = mix(base, asciiFg, ascii.a * asciiBlend);
    }

    f_color = vec4(clamp(base, 0.0, 1.0), 1.0);
}
