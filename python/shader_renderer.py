import pygame
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

import os
import platform

from hud import HUDOverlay, apply_display_adjustments

logger = logging.getLogger(__name__)


# ── ASCII character set ordered by visual density (sparse → dense) ──────────
# 10 character ramp is cleaner and less "muddy" for 3D extrusion
_ASCII_CHARS = " .:-=+*#%@"
_DENSITY_CHARS = list(_ASCII_CHARS)

# Glyph class lookup table (0: Micro, 1: Narrow, 2: Structural, 3: Enclosed)
_GLYPH_CLASSES = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]

# Atlas layout
_ATLAS_COLS = 10  # one column per character
_ATLAS_ROWS = 1
_CHAR_W = 8  # pixels per character cell width
_CHAR_H = 12  # pixels per character cell height

# Performance settings
TARGET_FPS = 60
MAX_TEXTURE_SIZE = (1920, 1080)  # Balance between quality and performance


def _build_ascii_atlas() -> bytes:
    """
    Render 16 ASCII characters into a horizontal strip texture.
    Returns raw RGBA bytes (width=160, height=16).
    The Alpha channel contains the visual glyph shape, while the Red channel encodes the Glyph Class.
    """
    import platform
    import sys

    atlas_w = _CHAR_W * _ATLAS_COLS
    atlas_h = _CHAR_H
    img = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to load a monospace font; fall back to default
    font = None
    system = platform.system()

    if system == "Windows":
        font_candidates = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/lucon.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            "/System/Library/Fonts/Monaco.dfont",
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Courier New.ttf",
        ]
    else:  # Linux
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/X11R6/lib/X11/fonts/misc/9x15.pcf.gz",
        ]

    # Also check common locations
    common_fonts = [
        "/usr/local/share/fonts/truetype/DejaVuSansMono.ttf",
    ]
    font_candidates.extend(common_fonts)

    for path in font_candidates:
        try:
            if os.path.exists(path):
                font = ImageFont.truetype(path, _CHAR_H - 2)
                break
        except Exception:
            pass

    if font is None:
        try:
            # Try to get default font
            font = ImageFont.load_default()
        except Exception:
            pass

    for i, ch in enumerate(_DENSITY_CHARS):
        x = i * _CHAR_W
        # Centre the glyph in its cell
        try:
            bbox = font.getbbox(ch)
            gw = bbox[2] - bbox[0]
            gh = bbox[3] - bbox[1]
            ox = x + (_CHAR_W - gw) // 2 - bbox[0]
            oy = (_CHAR_H - gh) // 2 - bbox[1]
        except Exception:
            ox, oy = x + 1, 1
            
        # Draw glyph in white to the alpha channel
        draw.text((ox, oy), ch, fill=(255, 255, 255, 255), font=font)
        
    # We need to bake the Glyph Class into the texture for the shader to read.
    # We will use the R channel for the glyph class (0.0 to 1.0) and A for the glyph itself.
    pixels = img.load()
    for i in range(_ATLAS_COLS):
        glyph_class = _GLYPH_CLASSES[i] / 3.0 # Map 0..3 to 0.0..1.0
        encoded_r = int(glyph_class * 255)
        for y in range(_CHAR_H):
            for x in range(i * _CHAR_W, (i + 1) * _CHAR_W):
                r, g, b, a = pixels[x, y]
                # If there is a pixel here, encode the class, otherwise keep it empty but classified
                pixels[x, y] = (encoded_r, 0, 0, a)

    return img.tobytes("raw", "RGBA")


class ShaderRenderer:

    # Crossfade duration in seconds
    CROSSFADE_DURATION = 0.35
    # ASCII mode — always on as focal point, no cycling
    ASCII_PERIOD = 25.0       # kept for _compute_ascii_blend compat
    ASCII_ON_FRACTION = 0.0   # DISABLED � causes abstract pattern feedback loop
    ASCII_FADE = 0.5          # fast fade (effectively unused at 1.0)

    # Performance monitoring
    def __init__(self, width=None, height=None, font_size=12):  # noqa: ARG002
        self._hud: HUDOverlay | None = None   # set after GL context is ready
        pygame.display.set_caption("V18")
        try:
            icon = pygame.image.load(os.path.join(os.path.dirname(__file__), "favicon.png"))
            pygame.display.set_icon(icon)
        except Exception as e:
            logger.warning(f"Could not load window icon: {e}")

        # Default to 1080p for better performance, with option for 4K
        self.width, self.height = 1920, 1080
        if width and height:
            self.width, self.height = (
                min(width, MAX_TEXTURE_SIZE[0]),
                min(height, MAX_TEXTURE_SIZE[1]),
            )

        logger.info(
            f"[ShaderRenderer] Creating windowed display: {self.width}x{self.height}"
        )

        try:
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
            )
        except pygame.error as e:
            logger.warning(
                f"Failed to create OpenGL display: {e}. Falling back to basic mode."
            )
            self.screen = pygame.display.set_mode(
                (self.width, self.height)
            )

        # Verify size
        self.width, self.height = self.screen.get_size()
        logger.info(f"[ShaderRenderer] Actual display size: {self.width}x{self.height}")

        try:
            self.ctx = moderngl.create_context()
            self._moderngl_available = True
        except Exception as e:
            logger.warning(
                f"ModernGL initialization failed: {e}. Running in fallback mode."
            )
            self._moderngl_available = False
            self.ctx = None

        if not self._moderngl_available:
            self._setup_fallback_renderer()
            return

        self._setup_shaders()
        self._setup_textures()
        self._setup_state_variables()

        # HUD overlay — created last so GL context and fonts are fully ready
        try:
            self._hud = HUDOverlay(self.ctx, self.width, self.height)
        except Exception as e:
            logger.warning(f"[ShaderRenderer] HUD init failed: {e}")
            self._hud = None

    def _setup_fallback_renderer(self):
        """Setup fallback renderer for systems without ModernGL support."""
        self._hud = None   # HUD requires ModernGL — unavailable in fallback
        self.quad_buffer = None
        self.program = None
        self.vao = None
        self.texture_a = None
        self.texture_b = None
        self.ascii_atlas = None
        self._crossfade_duration = self.CROSSFADE_DURATION
        self._smoothed_bpm = 120.0
        self._fx_mode = 0.0
        self._s_fx_blend = 0.0
        self._fx_active = False
        self._fx_active_time = 0.0
        self._fx_cooldown = 0.0
        self._energy_history = []
        self._last_ai_fx_time = 0.0
        self._ai_fx_mode = 0.0
        self._ai_fx_intensity = 0.0
        self._s_ai_hue_shift = 0.0
        self._s_ai_sat_boost = 0.0
        self._s_ai_glow = 0.0
        self._s_ai_edge_width = 0.0
        self._s_ai_edge_glow = 0.0
        self._ai_ascii_force = -1.0
        self._a_is_current = True
        self.start_time = pygame.time.get_ticks() / 1000.0
        self._crossfade_start = None
        self._crossfade_value = 0.0
        self._s_rms = 0.0
        self._s_bass = 0.0
        self._s_high = 0.0
        self._s_beat_env = 0.0
        self._s_centroid = 0.5
        self._s_ascii_blend = 0.0
        self._s_audio_presence = 0.0

    def _setup_shaders(self):
        """Setup OpenGL shaders with proper error handling."""
        try:
            self.quad_buffer = self.ctx.buffer(
                np.array(
                    [
                        -1.0,
                        -1.0,
                        0.0,
                        0.0,
                        1.0,
                        -1.0,
                        1.0,
                        0.0,
                        -1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],
                    dtype="f4",
                ).tobytes()
            )

            self.program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_vert;
                    in vec2 in_uv;
                    out vec2 v_uv;
                    // No zoom uniform needed for vertices if we do feedback in fragment
                    void main() {
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                        v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
                    }
                """,
                fragment_shader="""

                    #version 330
                    uniform sampler2D u_texture_a;
                    uniform sampler2D u_texture_b;
                    uniform sampler2D u_ascii_atlas;
                    uniform sampler2D u_feedback_tex;
                    uniform float u_crossfade;
                    uniform float u_time;
                    uniform float u_rms;
                    uniform float u_bass;
                    uniform float u_beat;
                    uniform float u_audio_presence;
                    uniform float u_dmt_active;
                    uniform vec2 u_resolution;
                    uniform vec2 u_ascii_cells;

                    in  vec2 v_uv;
                    out vec4 f_color;

                    vec3 rgb2hsv(vec3 c) {
                        vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
                        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                        float d = q.x - min(q.w, q.y);
                        float e = 1.0e-10;
                        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                    }

                    vec3 hsv2rgb(vec3 c) {
                        vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }

                    const float STEPS = 10.0;
                    const float MAX_EXTRUSION = 0.08; 
                    const vec2 EXTRUSION_DIR = normalize(vec2(-1.0, 1.0));

                    float getLuma(vec3 c) {
                        float l = dot(c, vec3(0.299, 0.587, 0.114));
                        return smoothstep(0.1, 0.9, l);
                    }

                    void main() {
                        vec2 uv = v_uv;
                        
                        vec3 col_a = texture(u_texture_a, uv).rgb;
                        vec3 col_b = texture(u_texture_b, uv).rgb;
                        float cf = smoothstep(0.0, 1.0, u_crossfade);
                        vec3 final_color = mix(col_a, col_b, cf);
                        
                        if (u_dmt_active < 0.5) {
                            vec3 hsv = rgb2hsv(final_color);
                            hsv.y = clamp(hsv.y * 1.3, 0.0, 1.0);
                            hsv.z = clamp(hsv.z * 1.05, 0.0, 1.0);
                            final_color = hsv2rgb(hsv);
                            f_color = vec4(final_color, 1.0);
                            return;
                        }

                        vec3 base_color = final_color;
                        float target_depth = getLuma(base_color);
                        
                        vec2 texel_size = vec2(u_ascii_cells.x / u_resolution.x, u_ascii_cells.y / u_resolution.y);
                        vec2 cell_uv = floor(uv / texel_size) * texel_size;
                        vec2 local_uv = fract(uv / texel_size);
                        
                        vec3 cell_color = mix(texture(u_texture_a, cell_uv).rgb, texture(u_texture_b, cell_uv).rgb, cf);
                        float raw_cell_depth = getLuma(cell_color);

                        float prev_depth = texture(u_feedback_tex, cell_uv).a;
                        float cell_depth = mix(prev_depth, raw_cell_depth, 0.25);

                        float d_right = texture(u_feedback_tex, cell_uv + vec2(texel_size.x, 0)).a;
                        float d_bot = texture(u_feedback_tex, cell_uv + vec2(0, texel_size.y)).a;
                        float d_br = texture(u_feedback_tex, cell_uv + vec2(texel_size.x, texel_size.y)).a;
                        
                        float local_ao = 1.0;
                        if(cell_depth < d_right) local_ao -= 0.25 * (d_right - cell_depth);
                        if(cell_depth < d_bot) local_ao -= 0.25 * (d_bot - cell_depth);
                        if(cell_depth < d_br) local_ao -= 0.3 * (d_br - cell_depth);
                        local_ao = clamp(local_ao, 0.15, 1.0);

                        vec3 march_color = vec3(0.0);
                        float accumulated_alpha = 0.0;

                        for(float i = 0.0; i < STEPS; i++) {
                            float t = i / STEPS;
                            float step_alpha = 1.0 - t; 
                            
                            vec2 march_uv = uv + EXTRUSION_DIR * (t * MAX_EXTRUSION);
                            vec2 m_cell_uv = floor(march_uv / texel_size.xy) * texel_size.xy;
                            vec2 m_local_uv = fract(march_uv / texel_size.xy);
                            
                            vec3 m_color = mix(texture(u_texture_a, m_cell_uv).rgb, texture(u_texture_b, m_cell_uv).rgb, cf);
                            
                            float m_prev_depth = texture(u_feedback_tex, m_cell_uv).a;
                            float m_target_depth = getLuma(m_color);
                            float m_depth = mix(m_prev_depth, m_target_depth, 0.25);
                            
                            if(t <= m_depth) {
                                float glyph_index = floor(m_depth * 9.0);
                                vec2 atlas_uv = vec2((glyph_index + m_local_uv.x) / 10.0, 1.0 - m_local_uv.y);
                                vec4 atlas_sample = texture(u_ascii_atlas, atlas_uv);
                                
                                float glyph_alpha = atlas_sample.a;
                                float glyph_class = atlas_sample.r;
                                
                                float effective_depth = m_depth - (1.0 - glyph_class) * 0.15;
                                if(t > effective_depth) continue;

                                float edge_shadow = 1.0;
                                if(m_local_uv.x > 0.85) edge_shadow *= mix(1.0, 0.4, clamp((m_depth - d_right)*12.0, 0.0, 1.0));
                                if(m_local_uv.y > 0.85) edge_shadow *= mix(1.0, 0.4, clamp((m_depth - d_bot)*12.0, 0.0, 1.0));

                                vec3 block_color = m_color;
                                vec3 diffuse = block_color * local_ao * edge_shadow;
                                vec3 step_col = mix(diffuse * (1.0 - t*0.4), block_color, 0.3); 
                                
                                if(t < 0.015) {
                                    step_col += vec3(pow(m_depth, 3.0) * 0.3 * local_ao);
                                }

                                float alpha_contrib = glyph_alpha * step_alpha * (1.0 - accumulated_alpha);
                                march_color += step_col * alpha_contrib;
                                accumulated_alpha += alpha_contrib;
                                
                                if(accumulated_alpha > 0.95) break; 
                            }
                        }
                        
                        if(accumulated_alpha < 1.0) {
                            march_color += base_color * 0.1 * (1.0 - accumulated_alpha);
                        }
                        
                        vec3 hsv = rgb2hsv(march_color);
                        hsv.y = clamp(hsv.y * 1.3, 0.0, 1.0);
                        hsv.z = clamp(hsv.z * 1.05, 0.0, 1.0);
                        march_color = hsv2rgb(hsv);
                        
                        f_color = vec4(march_color, cell_depth);
                    }

""",
            )

            self.vao = self.ctx.simple_vertex_array(
                self.program, self.quad_buffer, "in_vert", "in_uv"
            )
            logger.info("[ShaderRenderer] Shaders compiled successfully")

        except Exception as e:
            logger.error(f"Shader compilation failed: {e}")
            self._moderngl_available = False
            self._setup_fallback_renderer()

    def _setup_textures(self):
        """Setup textures with proper error handling."""
        try:
            # Two textures for crossfading incoming AI video
            self.texture_a = self.ctx.texture((self.width, self.height), 3)
            self.texture_b = self.ctx.texture((self.width, self.height), 3)
            
            # Recreate ASCII setup
            atlas_data = _build_ascii_atlas()
            self.ascii_atlas = self.ctx.texture(
                (_ATLAS_COLS * _CHAR_W, _ATLAS_ROWS * _CHAR_H), 4, atlas_data
            )
            self.ascii_atlas.filter = (moderngl.NEAREST, moderngl.NEAREST)

            self.texture_a.use(0)
            self.texture_b.use(1)
            self.ascii_atlas.use(2)
            self._set_uniform("u_texture_a", 0)
            self._set_uniform("u_texture_b", 1)
            self._set_uniform("u_ascii_atlas", 2)
            self._set_uniform("u_resolution", (float(self.width), float(self.height)))
            self._set_uniform("u_ascii_cells", (float(_CHAR_W), float(_CHAR_H)))

            # Two FBOs for ping-pong feedback history (we need to retain this for temporal stability if we add it back)
            self.fbo_tex_a = self.ctx.texture((self.width, self.height), 3)
            self.fbo_a = self.ctx.framebuffer(self.fbo_tex_a)
            self.fbo_tex_b = self.ctx.texture((self.width, self.height), 3)
            self.fbo_b = self.ctx.framebuffer(self.fbo_tex_b)
            self.fbo_a_is_current = True
            
            # Start both textures as solid black
            black = bytes(self.width * self.height * 3)
            self.texture_a.write(black)
            self.texture_b.write(black)
            self.fbo_tex_a.write(black)
            self.fbo_tex_b.write(black)

            self._a_is_current = True
            logger.info("[ShaderRenderer] Textures and FBOs initialized successfully")

        except Exception as e:
            logger.error(f"Texture setup failed: {e}")
            self._moderngl_available = False

    def _setup_state_variables(self):
        """Initialize all state variables."""
        self._crossfade_duration = self.CROSSFADE_DURATION
        self._smoothed_bpm = 120.0
        self._fx_mode = 0.0
        self._s_fx_blend = 0.0
        self._fx_active = False
        self._fx_active_time = 0.0
        self._fx_cooldown = 0.0
        self._energy_history = []
        self._last_ai_fx_time = pygame.time.get_ticks() / 1000.0
        self._ai_fx_mode = 0.0
        self._ai_fx_intensity = 0.0
        self._s_ai_hue_shift = 0.0
        self._s_ai_sat_boost = 0.0
        self._s_ai_glow = 0.0
        self._s_ai_edge_width = 0.0
        self._s_ai_edge_glow = 0.0
        self._ai_ascii_force = -1.0
        self.start_time = pygame.time.get_ticks() / 1000.0
        self._crossfade_start = None
        self._crossfade_value = 0.0
        self._s_rms = 0.0
        self._s_bass = 0.0
        self._s_high = 0.0
        self._s_beat_env = 0.0
        self._s_centroid = 0.5
        self._s_audio_presence = 0.0
        
        # Disable DMT feedback by default
        self._fx_dmt_active = False
        self._set_uniform("u_dmt_active", 0.0)

        # Borderless toggle state
        self._is_borderless = False
        self._last_click_ms = 0
        self._last_toggle_ms = 0
        self._ignoring_resize = False

    def _set_uniform(self, name: str, value):
        """Set a GLSL uniform by name, silently ignoring missing uniforms."""
        try:
            if self.program and name in self.program:
                self.program[name].value = value
        except Exception:
            pass

    def apply_ai_fx(self, fx_state: dict):
        """
        Accept FX directives from the AI agent.
        The user requested to remove all AI FX and rely only on the manual DMT trails.
        """
        pass

    def _to_bytes(self, image):
        # Normalise to numpy uint8 RGB array for uniform post-processing
        if isinstance(image, np.ndarray):
            if image.shape[:2] != (self.height, self.width):
                import cv2
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            arr = image
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.size != (self.width, self.height):
                image = image.resize((self.width, self.height), Image.NEAREST)
            arr = np.array(image)

        # Apply HUD display adjustments (brightness / contrast / sharpen / OLED)
        if self._hud is not None:
            arr = apply_display_adjustments(
                arr,
                self._hud.brightness,
                self._hud.contrast,
                self._hud.sharpen,
                self._hud.oled,
            )

        return arr.tobytes()

    def _start_crossfade(self, new_image):
        data = self._to_bytes(new_image)
        now = pygame.time.get_ticks() / 1000.0

        if self._a_is_current:
            self.texture_b.use(1)
            self.texture_b.write(data)
            self._crossfade_value = 0.0
            self._a_is_current = False
        else:
            self.texture_a.use(0)
            self.texture_a.write(data)
            self._crossfade_value = 1.0
            self._a_is_current = True

        self._crossfade_start = now

    def _compute_ascii_blend(self, t):
        """
        Returns target ascii blend 0-1 based on a periodic cycle.
        Waveform: ramp up, hold, ramp down, long rest.
        """
        phase = (t % self.ASCII_PERIOD) / self.ASCII_PERIOD  # 0..1
        on_frac = self.ASCII_ON_FRACTION
        fade_frac = self.ASCII_FADE / self.ASCII_PERIOD

        # ASCII is active from phase [0.5 - on_frac/2 .. 0.5 + on_frac/2]
        # centred at phase 0.5
        start = 0.5 - on_frac / 2.0
        end = 0.5 + on_frac / 2.0

        if phase < start:
            return 0.0
        elif phase < start + fade_frac:
            # Fade in
            return smoothstep_py((phase - start) / fade_frac)
        elif phase < end - fade_frac:
            # Full ascii
            return 1.0
        elif phase < end:
            # Fade out
            return smoothstep_py(1.0 - (phase - (end - fade_frac)) / fade_frac)
        else:
            return 0.0

    def _toggle_borderless(self):
        """Toggle borderless fullscreen using desktop resolution.
        On Windows, set_mode with OPENGL destroys the GL context — must fully
        recreate moderngl context, shaders and textures after every mode switch.
        """
        # Debounce: ignore if toggled less than 500ms ago
        now_ms = pygame.time.get_ticks()
        if now_ms - getattr(self, "_last_toggle_ms", 0) < 500:
            return
        self._last_toggle_ms = now_ms

        self._is_borderless = not self._is_borderless
        if self._is_borderless:
            info = pygame.display.Info()
            self.width, self.height = info.current_w, info.current_h
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.NOFRAME
        else:
            self.width, self.height = 1920, 1080
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE

        # set_mode invalidates the GL context on Windows — recreate everything
        self._ignoring_resize = True   # suppress the VIDEORESIZE event that follows
        self.screen = pygame.display.set_mode((self.width, self.height), flags)

        try:
            self.ctx = moderngl.create_context()
            self._moderngl_available = True
            self._setup_shaders()
            self._setup_textures()
            self._set_uniform("u_resolution", (float(self.width), float(self.height)))
            self._set_uniform("u_ascii_cells", (float(_CHAR_W), float(_CHAR_H)))
            self._set_uniform("u_dmt_active", 1.0 if self._fx_dmt_active else 0.0)
            logger.info(f"[ShaderRenderer] Borderless={'on' if self._is_borderless else 'off'} — GL context recreated {self.width}x{self.height}")
        except Exception as e:
            logger.warning(f"[ShaderRenderer] GL context recreation failed: {e}")

        if self._hud is not None:
            try:
                self._hud.resize(self.width, self.height)
            except Exception:
                pass

    def render(self, image, audio=None, status=None):
        if self._hud is not None and status is not None:
            self._hud.engine_status = status
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_f:
                    self._fx_dmt_active = not self._fx_dmt_active
                    self._set_uniform("u_dmt_active", 1.0 if self._fx_dmt_active else 0.0)
                elif event.key == pygame.K_h:
                    if self._hud is not None:
                        self._hud.visible = not self._hud.visible
                elif event.key == pygame.K_F11:
                    self._toggle_borderless()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Let HUD consume the click first (sliders / checkbox)
                if self._hud is not None and self._hud.on_mouse_down(
                    event.pos[0], event.pos[1]
                ):
                    pass  # event consumed — do NOT update last_click_ms
                else:
                    # Double-click outside HUD → toggle borderless fullscreen
                    now_ms = pygame.time.get_ticks()
                    if now_ms - self._last_click_ms < 400:
                        self._toggle_borderless()
                    self._last_click_ms = now_ms

            elif event.type == pygame.MOUSEMOTION:
                # Feed motion to HUD for slider dragging
                if self._hud is not None:
                    self._hud.on_mouse_motion(
                        event.pos[0], event.pos[1], event.buttons
                    )

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # Finalise drag and persist settings
                if self._hud is not None:
                    self._hud.on_mouse_up(event.pos[0], event.pos[1])

            elif event.type == pygame.VIDEORESIZE:
                # Ignore the resize event that fires immediately after our own set_mode call
                if self._ignoring_resize:
                    self._ignoring_resize = False
                    continue
                nw, nh = event.size
                if nw > 100 and nh > 100:
                    try:
                        self.width, self.height = nw, nh
                        self.screen = pygame.display.set_mode(
                            (self.width, self.height),
                            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
                        )
                        if self.ctx:
                            self.ctx.viewport = (0, 0, self.width, self.height)
                            self._setup_textures()
                        if self._hud is not None:
                            self._hud.resize(self.width, self.height)
                    except Exception as e:
                        logger.warning(f"Resize failed: {e}")

        if image is not None:
            self._start_crossfade(image)

        now = pygame.time.get_ticks() / 1000.0
        if self._crossfade_start is not None:
            elapsed = now - self._crossfade_start
            t = min(elapsed / self._crossfade_duration, 1.0)
            if self._a_is_current:
                self._crossfade_value = 1.0 - t
            else:
                self._crossfade_value = t

        # Fallback mode - simple rendering without shaders
        if not self._moderngl_available:
            return self._render_fallback(image)

        # Process audio data with proper smoothing
        if audio:
            audio_ema = 0.85
            rms = float(audio.get("smoothed_rms", 0.0)) / 5000.0
            self._s_rms = self._s_rms * audio_ema + rms * (1 - audio_ema)

            bass = float(audio.get("bass", 0.0)) / 2000.0
            self._s_bass = self._s_bass * audio_ema + bass * (1 - audio_ema)

            high = float(audio.get("high", 0.0)) / 1000.0
            self._s_high = self._s_high * audio_ema + high * (1 - audio_ema)

            beat = float(audio.get("beat_strength", 0.0))
            self._s_beat_env = self._s_beat_env * 0.9 + beat * 0.1

            centroid = float(audio.get("centroid", 0.5))
            self._s_centroid = self._s_centroid * audio_ema + centroid * (1 - audio_ema)

            # Audio presence detection
            if self._s_rms > 0.05:
                self._s_audio_presence = min(1.0, self._s_audio_presence + 0.05)
            else:
                self._s_audio_presence = max(0.0, self._s_audio_presence - 0.02)

        current_time = now - self.start_time

        # Set time and crossfade uniforms
        self._set_uniform("u_time", current_time)
        self._set_uniform("u_crossfade", self._crossfade_value)

        # Audio uniforms - now active with proper smoothing
        self._set_uniform("u_rms", self._s_rms)
        self._set_uniform("u_bass", self._s_bass)
        self._set_uniform("u_beat", self._s_beat_env)
        self._set_uniform("u_audio_presence", self._s_audio_presence)

        # Set up Ping Pong FBO for feedback trails
        if getattr(self, "fbo_a_is_current", True):
            target_fbo = self.fbo_a
            self.fbo_tex_b.use(3) # Sample from the OTHER FBO to get feedback
        else:
            target_fbo = self.fbo_b
            self.fbo_tex_a.use(3)
        
        self._set_uniform("u_feedback_tex", 3)

        # 1. Render the Shader into the Target FBO
        target_fbo.use()
        self.texture_a.use(0)
        self.texture_b.use(1)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # 2. Copy FBO to the Pygame screen
        self.ctx.screen.use()
        self.ctx.copy_framebuffer(self.ctx.screen, target_fbo)

        # 3. Blend HUD overlay on top (no-op when not visible)
        if self._hud is not None:
            self._hud.render()

        pygame.display.flip()

        # Swap FBOs for the next frame
        if hasattr(self, "fbo_a_is_current"):
            self.fbo_a_is_current = not self.fbo_a_is_current

        return True

    def _render_fallback(self, image, audio=None):  # noqa: ARG002
        """Simple fallback rendering without ModernGL shaders."""
        self.screen.fill((0, 0, 0))
        if image is not None:
            # Simple image blit without shaders
            try:
                if isinstance(image, np.ndarray):
                    img = Image.fromarray(image)
                else:
                    img = image.copy()
                if img.size != self.screen.get_size():
                    img = img.resize(self.screen.get_size(), Image.LANCZOS)
                surface = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
                self.screen.blit(surface, (0, 0))
            except Exception as e:
                logger.warning(f"Fallback render error: {e}")
        pygame.display.flip()
        return True

    def stop(self):
        if self._hud is not None:
            self._hud.release()
        pygame.quit()


def smoothstep_py(x: float) -> float:
    """Python-side smoothstep for ASCII blend curve."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


if __name__ == "__main__":
    renderer = ShaderRenderer()
    test_image = Image.new("RGB", (1280, 720), (100, 50, 200))
    running = True
    while running:
        running = renderer.render(test_image)
    renderer.stop()
