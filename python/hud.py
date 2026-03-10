"""
HUD Overlay — display control centre drawn as an OpenGL alpha-blended quad.

Toggle visibility: H key (handled in shader_renderer.py event loop).

Controls
--------
  Brightness  -1.0 .. +1.0   (±100 intensity points, shown as ±100 %)
  Contrast     0.5 .. 2.0    (multiplied around midpoint 128, shown as ×value)
  Sharpening   0.0 .. 1.0    (unsharp-mask strength, shown as 0–100 %)
  OLED         bool           OLED-mode: crush near-blacks, boost whites slightly

Settings are persisted to  data/hud_settings.json  and reloaded on start.

Rendering strategy
------------------
The HUD panel is drawn each frame onto a small pygame SRCALPHA surface using
pygame.draw primitives, then uploaded as an RGBA OpenGL texture.  A minimal
fullscreen-quad shader blends the texture over the rendered frame using standard
src-alpha compositing.  This keeps all HUD drawing on the CPU (fast enough at
the ~340×300 panel size) while compositing on the GPU with zero copy overhead.
"""

from __future__ import annotations

import json
import os

import moderngl
import numpy as np
import pygame
import pygame.font

# ── Layout constants ──────────────────────────────────────────────────────────
_PANEL_W  = 340     # panel pixel width
_PANEL_H  = 305     # panel pixel height
_MARGIN   = 22      # gap from right/top screen edge

_PAD_X    = 12      # left padding inside panel
_SLIDER_X = 90      # slider track left edge (panel-relative)
_SLIDER_W = 210     # slider track width
_SLIDER_H = 8       # slider track height
_ROW_H    = 66      # vertical spacing between slider rows
_FIRST_Y  = 68      # Y of first slider track inside panel

# Colour palette (RGBA tuples — pygame draw functions accept 3- or 4-tuples)
_C_BG       = (14,  16,  22,  218)
_C_BORDER   = (65, 110, 165, 255)
_C_TITLE    = (155, 205, 255, 255)
_C_DIVIDER  = (50,  82, 128, 255)
_C_LABEL    = (165, 192, 212, 255)
_C_TRACK    = (38,  44,  57,  255)
_C_FILL     = (52, 118, 208, 255)
_C_HANDLE   = (200, 220, 255, 255)
_C_HANDLE_B = (88, 152, 252, 255)
_C_VALUE    = (125, 170, 210, 255)
_C_CB_ON    = (78, 215, 138, 255)
_C_CB_FRAME = (75, 110, 160, 255)
_C_KEY_HINT = (100, 130, 160, 255)


# ── Post-processing helper ────────────────────────────────────────────────────

def apply_display_adjustments(
    img: np.ndarray,
    brightness: float,
    contrast: float,
    sharpen: float,
    oled: bool,
) -> np.ndarray:
    """
    Apply display post-processing to a uint8 RGB numpy array (H, W, 3).

    All operations are performed in float32 and clipped back to uint8.

    Parameters
    ----------
    img        : (H, W, 3) uint8 ndarray
    brightness : -1.0 .. +1.0  — additive offset (±255 points)
    contrast   : 0.5  .. 2.0   — multiplicative scale around midpoint 128
    sharpen    : 0.0  .. 1.0   — unsharp-mask strength
    oled       : bool           — OLED black-crush + white-boost

    Returns
    -------
    uint8 ndarray, same shape.
    """
    # Fast-exit: nothing to do
    if brightness == 0.0 and contrast == 1.0 and sharpen == 0.0 and not oled:
        return img

    out = img.astype(np.float32)

    # Brightness — additive offset across all channels
    if brightness != 0.0:
        out += brightness * 255.0

    # Contrast — scale around perceptual midpoint
    if contrast != 1.0:
        out = 128.0 + (out - 128.0) * contrast

    # OLED optimisation
    # • Crush near-blacks to true black  (pixel < threshold → 0)
    # • Slight boost to near-white      (pixel > 235 → ×1.05, capped at 255)
    if oled:
        black_mask = out < 18.0
        out[black_mask] = 0.0
        white_mask = out > 235.0
        out[white_mask] = np.minimum(255.0, out[white_mask] * 1.05)

    np.clip(out, 0.0, 255.0, out=out)
    out = out.astype(np.uint8)

    # Sharpening — unsharp mask (cv2 for speed; PIL fallback)
    if sharpen > 0.01:
        strength = sharpen * 2.0        # map 0–1 → 0–2 addWeighted param
        try:
            import cv2
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
            out = cv2.addWeighted(out, 1.0 + strength, blurred, -strength, 0.0)
            np.clip(out, 0, 255, out=out)
        except ImportError:
            from PIL import Image, ImageFilter
            pct = int(sharpen * 150)
            pil = Image.fromarray(out).filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=pct, threshold=3)
            )
            out = np.array(pil)

    return out


# ── HUD overlay class ─────────────────────────────────────────────────────────

class HUDOverlay:
    """
    Renders a semi-transparent control panel as an OpenGL overlay.

    Usage
    -----
        hud = HUDOverlay(ctx, screen_w, screen_h)
        # in render loop:
        hud.render()          # no-op when not visible; draws + uploads each frame
        pygame.display.flip()
    """

    def __init__(self, ctx: moderngl.Context, screen_w: int, screen_h: int) -> None:
        self.ctx       = ctx
        self.screen_w  = screen_w
        self.screen_h  = screen_h
        self.visible   = False

        # ── Control values ────────────────────────────────────────────────────
        self.brightness: float = 0.0    # -1.0 .. +1.0
        self.contrast:   float = 1.0    # 0.5  .. 2.0
        self.sharpen:    float = 0.0    # 0.0  .. 1.0
        self.oled:       bool  = False
        self.spout:      bool  = True
        self.engine_status: str = "WAITING FOR FRAMES..."

        # ── Drag state ────────────────────────────────────────────────────────
        # Tuple (key, vmin, vmax, track_x, track_w) while dragging a slider
        self._drag: tuple | None = None

        # ── Panel anchor (top-right of screen) ────────────────────────────────
        self.px = screen_w - _PANEL_W - _MARGIN
        self.py = _MARGIN

        # ── Pygame drawing surface ────────────────────────────────────────────
        pygame.font.init()
        self._surf = pygame.Surface((_PANEL_W, _PANEL_H), pygame.SRCALPHA)
        self._load_fonts()

        # ── OpenGL resources ──────────────────────────────────────────────────
        # RGBA texture — same size as the panel surface
        self._tex = ctx.texture((_PANEL_W, _PANEL_H), 4)
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Minimal overlay shader: just sample texture and output with alpha
        self._prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_hud;
                in  vec2 v_uv;
                out vec4 f_color;
                void main() {
                    f_color = texture(u_hud, v_uv);
                }
            """,
        )
        self._prog["u_hud"].value = 7   # dedicated texture unit

        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._build_quad()

        # ── Persistence ───────────────────────────────────────────────────────
        _data_dir = os.path.join(os.path.dirname(__file__), "data")
        self._settings_path = os.path.join(_data_dir, "hud_settings.json")
        self._load_settings()

    # ── Font loading ─────────────────────────────────────────────────────────

    def _load_fonts(self) -> None:
        for name in ("consolas", "courier new", "lucida console", "courier"):
            try:
                self._font_title = pygame.font.SysFont(name, 14, bold=True)
                self._font_label = pygame.font.SysFont(name, 12)
                self._font_val   = pygame.font.SysFont(name, 11)
                return
            except Exception:
                continue
        # Fallback to pygame built-in bitmap font
        self._font_title = pygame.font.Font(None, 18)
        self._font_label = pygame.font.Font(None, 16)
        self._font_val   = pygame.font.Font(None, 14)

    # ── OpenGL quad ───────────────────────────────────────────────────────────

    def _build_quad(self) -> None:
        """Rebuild the NDC quad covering the HUD panel after a resize."""
        sw, sh = self.screen_w, self.screen_h
        x0, y0 = self.px, self.py
        x1, y1 = x0 + _PANEL_W, y0 + _PANEL_H

        # Screen → NDC  (screen Y=0 at top → NDC Y=+1)
        nx0 =  x0 / sw * 2.0 - 1.0
        nx1 =  x1 / sw * 2.0 - 1.0
        ny0 =  1.0 - y0 / sh * 2.0   # NDC y for panel TOP  (higher value)
        ny1 =  1.0 - y1 / sh * 2.0   # NDC y for panel BOTTOM (lower value)

        # We upload the pygame surface with flip=True so the texture is stored
        # bottom-row-first (OpenGL convention).  UV(0,1) → panel top,
        # UV(0,0) → panel bottom.  TRIANGLE_STRIP order: BL, BR, TL, TR.
        verts = np.array([
            nx0, ny1,  0.0, 0.0,   # screen bottom-left  → tex UV(0,0)
            nx1, ny1,  1.0, 0.0,   # screen bottom-right → tex UV(1,0)
            nx0, ny0,  0.0, 1.0,   # screen top-left     → tex UV(0,1)
            nx1, ny0,  1.0, 1.0,   # screen top-right    → tex UV(1,1)
        ], dtype="f4")

        if self._vbo is not None:
            self._vao.release()
            self._vbo.release()
        self._vbo = self.ctx.buffer(verts.tobytes())
        self._vao = self.ctx.simple_vertex_array(
            self._prog, self._vbo, "in_pos", "in_uv"
        )

    # ── Slider geometry helpers ───────────────────────────────────────────────

    def _slider_track(self, idx: int) -> tuple[int, int, int, int]:
        """Return (tx, ty, tw, th) of slider track — panel-relative coords."""
        return _SLIDER_X, _FIRST_Y + idx * _ROW_H, _SLIDER_W, _SLIDER_H

    def _val_to_hx(self, val: float, vmin: float, vmax: float,
                   tx: int, tw: int) -> int:
        t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
        return int(tx + t * tw)

    def _hx_to_val(self, hx: int, vmin: float, vmax: float,
                   tx: int, tw: int) -> float:
        t = max(0.0, min(1.0, (hx - tx) / tw))
        return vmin + t * (vmax - vmin)

    # ── Panel drawing ─────────────────────────────────────────────────────────

    def draw(self) -> None:
        """Redraw the HUD surface and upload to the GPU texture."""
        surf = self._surf
        surf.fill((0, 0, 0, 0))

        # Background panel + border
        pygame.draw.rect(surf, _C_BG,     (0, 0, _PANEL_W, _PANEL_H), border_radius=10)
        pygame.draw.rect(surf, _C_BORDER, (0, 0, _PANEL_W, _PANEL_H), 1, border_radius=10)

        # Title bar
        title = self._font_title.render(" DISPLAY SETTINGS", True, _C_TITLE[:3])
        surf.blit(title, (_PAD_X, 10))
        hint = self._font_val.render("[H] to hide", True, _C_KEY_HINT[:3])
        surf.blit(hint, (_PANEL_W - hint.get_width() - _PAD_X, 12))
        
        # Status Line
        status_color = (255, 180, 50) if "WAITING" in self.engine_status or "WARMING" in self.engine_status else (100, 255, 100)
        status_surf = self._font_val.render(f"STATUS: {self.engine_status}", True, status_color)
        surf.blit(status_surf, (_PAD_X, 32))

        pygame.draw.line(surf, _C_DIVIDER, (_PAD_X, 48), (_PANEL_W - _PAD_X, 48))

        # ── Sliders ───────────────────────────────────────────────────────────
        _sliders = [
            ("BRIGHTNESS", self.brightness, -1.0, 1.0,  "brightness"),
            ("CONTRAST",   self.contrast,    0.5, 2.0,  "contrast"),
            ("SHARPENING", self.sharpen,     0.0, 1.0,  "sharpen"),
        ]
        for i, (lbl, val, vmin, vmax, key) in enumerate(_sliders):
            tx, ty, tw, th = self._slider_track(i)

            # Row label
            lbl_surf = self._font_label.render(lbl, True, _C_LABEL[:3])
            surf.blit(lbl_surf, (_PAD_X, ty - 16))

            # Track background
            pygame.draw.rect(surf, _C_TRACK[:3], (tx, ty, tw, th), border_radius=4)

            # Filled portion (left of handle)
            hx = self._val_to_hx(val, vmin, vmax, tx, tw)
            fill_w = max(2, hx - tx)
            pygame.draw.rect(surf, _C_FILL[:3], (tx, ty, fill_w, th), border_radius=4)

            # Handle
            hy = ty + th // 2
            pygame.draw.circle(surf, _C_HANDLE[:3],   (hx, hy), 9)
            pygame.draw.circle(surf, _C_HANDLE_B[:3], (hx, hy), 9, 2)

            # Value label (right-aligned)
            if key == "brightness":
                vs = f"{val * 100:+.0f}%"
            elif key == "contrast":
                vs = f"{val:.2f}\u00d7"
            else:
                vs = f"{val * 100:.0f}%"
            vsurf = self._font_val.render(vs, True, _C_VALUE[:3])
            surf.blit(vsurf, (_PANEL_W - _PAD_X - vsurf.get_width(), ty - 3))

        # ── OLED checkbox ─────────────────────────────────────────────────────
        cb_y = _FIRST_Y + 3 * _ROW_H - 10
        pygame.draw.line(surf, _C_DIVIDER,
                         (_PAD_X, cb_y - 8), (_PANEL_W - _PAD_X, cb_y - 8))
        cb_rect = pygame.Rect(_PAD_X, cb_y, 16, 16)
        pygame.draw.rect(surf, _C_TRACK[:3],   cb_rect, border_radius=3)
        pygame.draw.rect(surf, _C_CB_FRAME[:3], cb_rect, 1, border_radius=3)
        if self.oled:
            # Checkmark
            pts = [
                (cb_rect.x + 3,  cb_rect.y + 8),
                (cb_rect.x + 7,  cb_rect.y + 12),
                (cb_rect.x + 13, cb_rect.y + 4),
            ]
            pygame.draw.lines(surf, _C_CB_ON[:3], False, pts, 2)
        oled_lbl = self._font_label.render("OLED Optimization", True, _C_LABEL[:3])
        surf.blit(oled_lbl, (_PAD_X + 22, cb_y + 1))

        # ── Spout checkbox ────────────────────────────────────────────────────
        cb_y2 = cb_y + 24
        cb_rect2 = pygame.Rect(_PAD_X, cb_y2, 16, 16)
        pygame.draw.rect(surf, _C_TRACK[:3],   cb_rect2, border_radius=3)
        pygame.draw.rect(surf, _C_CB_FRAME[:3], cb_rect2, 1, border_radius=3)
        if self.spout:
            # Checkmark
            pts = [
                (cb_rect2.x + 3,  cb_rect2.y + 8),
                (cb_rect2.x + 7,  cb_rect2.y + 12),
                (cb_rect2.x + 13, cb_rect2.y + 4),
            ]
            pygame.draw.lines(surf, _C_CB_ON[:3], False, pts, 2)
        spout_lbl = self._font_label.render("Enable Spout/Syphon Output", True, _C_LABEL[:3])
        surf.blit(spout_lbl, (_PAD_X + 22, cb_y2 + 1))

        # Upload to GPU.  flip=True: pygame top row → OpenGL UV.y=1 (correct)
        raw = pygame.image.tobytes(surf, "RGBA", True)
        self._tex.write(raw)

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self) -> None:
        """
        Blend the HUD quad over the current screen contents.
        Call after the main content is on screen, before display.flip().
        No-op when not visible.
        """
        if not self.visible:
            return

        self.draw()                                         # always refresh
        self._tex.use(7)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.screen.use()
        self._vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    # ── Resize ────────────────────────────────────────────────────────────────

    def resize(self, w: int, h: int) -> None:
        self.screen_w, self.screen_h = w, h
        self.px = w - _PANEL_W - _MARGIN
        self._build_quad()

    # ── Hit testing ───────────────────────────────────────────────────────────

    def contains(self, sx: int, sy: int) -> bool:
        """True if screen point (sx, sy) is within the HUD panel."""
        return (
            self.visible
            and self.px <= sx < self.px + _PANEL_W
            and self.py <= sy < self.py + _PANEL_H
        )

    # ── Input handling ────────────────────────────────────────────────────────

    def on_mouse_down(self, sx: int, sy: int) -> bool:
        """
        Handle MOUSEBUTTONDOWN.  Returns True if the event was consumed,
        preventing it from reaching other handlers (double-click, etc.).
        """
        if not self.visible:
            return False

        lx, ly = sx - self.px, sy - self.py

        # Check sliders
        _slider_meta = [
            ("brightness", self.brightness, -1.0, 1.0, 0),
            ("contrast",   self.contrast,    0.5, 2.0, 1),
            ("sharpen",    self.sharpen,     0.0, 1.0, 2),
        ]
        for key, val, vmin, vmax, idx in _slider_meta:
            tx, ty, tw, th = self._slider_track(idx)
            hy = ty + th // 2
            hx = self._val_to_hx(val, vmin, vmax, tx, tw)
            in_track  = (tx - 5 <= lx <= tx + tw + 5) and (ty - 6 <= ly <= ty + th + 6)
            near_handle = abs(lx - hx) <= 14 and abs(ly - hy) <= 14
            if in_track or near_handle:
                self._drag = (key, vmin, vmax, tx, tw)
                self._apply_drag(lx)
                return True

        # Check OLED checkbox
        cb_y = _FIRST_Y + 3 * _ROW_H - 10
        if _PAD_X <= lx <= _PAD_X + 16 and cb_y <= ly <= cb_y + 16:
            self.oled = not self.oled
            self._save_settings()
            return True

        # Check Spout checkbox
        cb_y2 = cb_y + 24
        if _PAD_X <= lx <= _PAD_X + 16 and cb_y2 <= ly <= cb_y2 + 16:
            self.spout = not self.spout
            self._save_settings()
            return True

        # Absorb all clicks inside the panel (so they don't trigger borderless)
        if 0 <= lx < _PANEL_W and 0 <= ly < _PANEL_H:
            return True

        return False

    def on_mouse_motion(self, sx: int, sy: int, buttons: tuple) -> bool:
        """Handle MOUSEMOTION while dragging.  Returns True if consumed."""
        if self._drag and buttons[0]:
            self._apply_drag(sx - self.px)
            return True
        if not buttons[0]:
            self._drag = None
        return False

    def on_mouse_up(self, sx: int, sy: int) -> bool:
        """Handle MOUSEBUTTONUP.  Saves settings when drag finishes."""
        if self._drag:
            self._drag = None
            self._save_settings()
            return True
        return False

    def _apply_drag(self, lx: int) -> None:
        if not self._drag:
            return
        key, vmin, vmax, tx, tw = self._drag
        v = self._hx_to_val(lx, vmin, vmax, tx, tw)
        if key == "brightness":
            self.brightness = v
        elif key == "contrast":
            self.contrast = v
        elif key == "sharpen":
            self.sharpen = v

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_settings(self) -> None:
        try:
            if os.path.exists(self._settings_path):
                with open(self._settings_path) as f:
                    d = json.load(f)
                self.brightness = float(d.get("brightness", 0.0))
                self.contrast   = float(d.get("contrast",   1.0))
                self.sharpen    = float(d.get("sharpen",    0.0))
                self.oled       = bool(d.get("oled",        False))
                self.spout      = bool(d.get("spout",       True))
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            tmp = self._settings_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "brightness": self.brightness,
                        "contrast":   self.contrast,
                        "sharpen":    self.sharpen,
                        "oled":       self.oled,
                        "spout":      self.spout,
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp, self._settings_path)
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Free OpenGL resources."""
        try:
            if self._vao:
                self._vao.release()
            if self._vbo:
                self._vbo.release()
            self._tex.release()
            self._prog.release()
        except Exception:
            pass
