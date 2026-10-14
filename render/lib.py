"""Shared 2D compositing helpers for the shenhai MV drafts.

Images are float32 sRGB in [0, 1]. Layers are premultiplied RGBA (H, W, 4).
"""
import math
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 24
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "source"                # the original character art: deepseekgirl.png, deepseekgirljianyhing.png
MUSIC = ROOT / "audio" / "music11_edit.wav"     # the film's edit of music11.mp3 (tools/edit_audio.py)
PREVIEW = ROOT / "drafts" / "preview_full_v2.mp4"   # the joined film: later nodes flash back to frames of earlier ones
PREVIEW_LEAD = 3.0                                  # the joined film opens with the 3 s notice (render/intro.py)
FONT_SERIF = "C:/Windows/Fonts/NotoSerifSC-VF.ttf"
FONT_SANS = "C:/Windows/Fonts/NotoSansSC-VF.ttf"

# Character landmarks in deepseekgirl.png full-res pixels (3072x5428)
CHAR = dict(top=(1444, 230), face=(1500, 880), eye_l=(1360, 825), eye_r=(1635, 745),
            mouth=(1530, 945), waist=(1444, 1930), hand=(2600, 1330), center=(1444, 2700),
            hem=(1444, 4608), feet=(1444, 5410))


def film_frame(t, path):
    """Save the joined film's frame at song time t (edit timeline) to path, once. Flashbacks are made of these."""
    path = Path(path)
    if not path.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t + PREVIEW_LEAD:.3f}", "-i", str(PREVIEW), "-frames:v", "1",
                        str(path)], check=True)
    return path


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


PAL = dict(abyss=hexc("#02060f"), deep=hexc("#061634"), mid=hexc("#123a7c"), blue=hexc("#2e64b8"),
           pale=hexc("#9fd4ff"), ice=hexc("#e8f7ff"), violet=hexc("#5b48c2"),
           orange=hexc("#ff9a3c"), ember=hexc("#ff6a1f"), amber=hexc("#ffc27a"),
           paper=hexc("#f1ebdf"), ink=hexc("#16213d"))


# ------------------------------------------------------------------ timing / easing
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def seg(t, a, b):
    return float(clamp01((t - a) / (b - a)))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    return 1 - (1 - clamp01(x)) ** p


def ease_in(x, p=3):
    return clamp01(x) ** p


def ease_back(x, k=1.6):
    x = clamp01(x) - 1
    return 1 + x * x * ((k + 1) * x + k)


def lerp(a, b, x):
    return a + (b - a) * x


def pulse(t, t0, decay=8.0):
    """Instant attack at t0, exponential decay after."""
    return math.exp(-(t - t0) * decay) if t >= t0 else 0.0


# ------------------------------------------------------------------ images / layers
def load_rgba(path, scale=1.0, alpha_gain=1.0):
    a = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255
    a[..., 3] = np.clip(a[..., 3] * alpha_gain, 0, 1)
    a[..., :3] *= a[..., 3:4]
    if scale != 1:
        a = cv2.resize(a, (round(a.shape[1] * scale), round(a.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return a


def load_rgb(path, size=None):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255
    if size:
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    return a


def affine(anchor, pos, scale, rot=0.0):
    """Matrix taking source px -> dest px: anchor lands on pos, scaled and rotated (deg, ccw on screen)."""
    c = math.cos(math.radians(-rot)) * scale
    s = math.sin(math.radians(-rot)) * scale
    M = np.array([[c, -s, 0], [s, c, 0]], np.float32)
    M[:, 2] = np.asarray(pos, np.float32) - M[:, :2] @ np.asarray(anchor, np.float32)
    return M


def warp(src, M, size=(W, H), interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(src, M, size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def apply_pt(M, p):
    return (float(M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]), float(M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]))


class Sprite:
    """Premultiplied RGBA with a few pre-scaled levels so big downscales stay clean."""

    def __init__(self, rgba, levels=(1.0, 0.5, 0.25, 0.125), base=1.0):
        self.base = base  # rgba is already `base` x the reference resolution
        self.levels = {}
        for lv in levels:
            f = lv / levels[0]
            self.levels[lv * base] = rgba if f == 1 else cv2.resize(
                rgba, (max(1, round(rgba.shape[1] * f)), max(1, round(rgba.shape[0] * f))), interpolation=cv2.INTER_AREA)

    def matrix(self, anchor, pos, scale, rot=0.0):
        cands = [lv for lv in self.levels if lv >= scale * 0.9]
        lv = min(cands) if cands else max(self.levels)
        return lv, affine((anchor[0] * lv, anchor[1] * lv), pos, scale / lv, rot)

    def render(self, anchor, pos, scale, rot=0.0, size=(W, H)):
        lv, M = self.matrix(anchor, pos, scale, rot)
        return warp(self.levels[lv], M, size)


def over(dst, layer):
    return layer[..., :3] + dst * (1 - layer[..., 3:4])


def with_alpha(layer, a):
    return layer * a


def solid_layer(mask, color, alpha=1.0):
    m = (mask * alpha)[..., None]
    return np.concatenate([m * color, m], axis=2).astype(np.float32)


def unpremult(layer):
    a = layer[..., 3:4]
    return np.where(a > 1e-4, layer[..., :3] / np.maximum(a, 1e-4), 0), a


def premult(rgb, a):
    return np.concatenate([rgb * a, a], axis=2).astype(np.float32)


def blur(img, sigma):
    if sigma <= 0.3:
        return img
    if sigma > 10:
        f = max(2, int(sigma // 5))
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rim(alpha, dx, dy, sigma=1.5):
    """Edge light on the side facing (-dx, -dy): alpha minus alpha shifted toward the light."""
    r = np.clip(alpha - shift(alpha, dx, dy), 0, 1)
    return blur(r, sigma)


def vgrad(h, w, top, bottom, gamma=1.0):
    y = (np.linspace(0, 1, h, dtype=np.float32) ** gamma)[:, None, None]
    return np.broadcast_to(top * (1 - y) + bottom * y, (h, w, 3)).astype(np.float32).copy()


def radial(h, w, cx, cy, rx, ry, power=2.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    return np.clip(1 - d, 0, 1) ** power


def grade(rgb, mul=(1, 1, 1), add=(0, 0, 0), sat=1.0, gamma=1.0):
    out = rgb * np.asarray(mul, np.float32) + np.asarray(add, np.float32)
    if sat != 1.0:
        lum = out.mean(axis=-1, keepdims=True)
        out = lum + (out - lum) * sat
    if gamma != 1.0:
        out = np.clip(out, 0, None) ** gamma
    return out


def grade_layer(layer, **kw):
    rgb, a = unpremult(layer)
    return premult(grade(rgb, **kw), a)


def posterize_layer(layer, levels=6):
    rgb, a = unpremult(layer)
    rgb = np.round(np.clip(rgb, 0, 1) * (levels - 1)) / (levels - 1)
    return premult(rgb, a)


def remap_wobble(layer, t, amp, freq=0.011, speed=1.4, xfreq=0.0035):
    """Underwater sway: horizontal displacement whose amplitude follows `amp` (array over y)."""
    h, w = layer.shape[:2]
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    dx = amp[:, None] * np.sin(ys * freq + t * speed + xs * xfreq)
    mx = (xs + dx).astype(np.float32)
    my = np.broadcast_to(ys, (h, w)).astype(np.float32)
    return cv2.remap(layer, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------------------ procedural textures
def value_noise(h, w, cell, seed=0, octaves=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = max(2, int(cell / (2 ** o)))
        g = rng.random((h // c + 3, w // c + 3)).astype(np.float32)
        up = cv2.resize(g, (g.shape[1] * c, g.shape[0] * c), interpolation=cv2.INTER_CUBIC)[:h, :w]
        out += up * amp
        tot += amp
        amp *= 0.5
    return out / tot


class Caustics:
    """Animated Voronoi edge network (F2-F1), evaluated at low res then upscaled."""

    def __init__(self, w, h, cell=60, seed=0, speed=0.7, low=4, warp=14.0):
        self.w, self.h, self.low = w, h, low
        self.lw, self.lh = w // low, h // low
        self.cell = cell / low
        rng = np.random.default_rng(seed)
        self.gx = int(math.ceil(self.lw / self.cell)) + 3
        self.gy = int(math.ceil(self.lh / self.cell)) + 3
        self.ph = rng.random((self.gy, self.gx, 2)).astype(np.float32) * 6.283
        self.sp = (0.6 + 0.8 * rng.random((self.gy, self.gx, 2))).astype(np.float32) * speed
        self.jit = (0.2 + 0.25 * rng.random((self.gy, self.gx))).astype(np.float32)
        ys, xs = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        nx = value_noise(self.lh, self.lw, 40 // low + 2, seed + 1, 2) - 0.5
        ny = value_noise(self.lh, self.lw, 40 // low + 2, seed + 2, 2) - 0.5
        self.xs = xs + nx * warp / low
        self.ys = ys + ny * warp / low

    def field(self, t):
        c = self.cell
        jx = 0.5 + self.jit * np.sin(t * self.sp[..., 0] + self.ph[..., 0])
        jy = 0.5 + self.jit * np.cos(t * self.sp[..., 1] + self.ph[..., 1])
        px = (np.arange(self.gx, dtype=np.float32)[None, :] - 1 + jx) * c
        py = (np.arange(self.gy, dtype=np.float32)[:, None] - 1 + jy) * c
        ci = np.clip((self.xs // c).astype(int) + 1, 1, self.gx - 2)
        cj = np.clip((self.ys // c).astype(int) + 1, 1, self.gy - 2)
        d1 = np.full(self.xs.shape, 1e9, np.float32)
        d2 = d1.copy()
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                X = px[cj + dj, ci + di]
                Y = py[cj + dj, ci + di]
                d = (self.xs - X) ** 2 + (self.ys - Y) ** 2
                d2 = np.minimum(d2, np.maximum(d1, d))
                d1 = np.minimum(d1, d)
        f = (np.sqrt(d2) - np.sqrt(d1)) / c
        return cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

    def lines(self, t, width=0.09, soft=0.05):
        f = self.field(t)
        return np.clip((width - f) / soft, 0, 1).astype(np.float32)

    def net(self, t, width=0.05):
        """Thin soft caustic filaments."""
        f = self.field(t)
        return np.exp(-(f / width) ** 2).astype(np.float32)

    def blobs(self, t, thresh=0.16):
        """Flat anime-style light patches (cells shrunk away from their borders)."""
        f = self.field(t)
        return np.clip((f - thresh) * 25, 0, 1).astype(np.float32)


def flowlines(w, h, angle, n=2500, steps=90, step=3.0, seed=0, width=1, ss=2, levels=(0.35, 0.55, 0.75, 1.0)):
    """Engraving-like streamlines following an angle field (h, w) in radians."""
    rng = np.random.default_rng(seed)
    P = rng.random((n, 2)).astype(np.float32) * np.array([w, h], np.float32)
    pts = np.empty((steps, n, 2), np.float32)
    for s in range(steps):
        xi = np.clip(P[:, 0].astype(int), 0, w - 1)
        yi = np.clip(P[:, 1].astype(int), 0, h - 1)
        a = angle[yi, xi]
        P = P + step * np.stack([np.cos(a), np.sin(a)], 1)
        pts[s] = P
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    lv = rng.integers(0, len(levels), n)
    for k, level in enumerate(levels):
        idx = np.nonzero(lv == k)[0]
        polys = [np.round(pts[:, i, :] * ss).astype(np.int32) for i in idx]
        cv2.polylines(canvas, polys, False, int(255 * level), thickness=width * ss, lineType=cv2.LINE_AA)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def make_glyphs(n=48, size=72, seed=3):
    """Invented 'rune' glyphs: strokes between points of a 3x4 lattice."""
    rng = np.random.default_rng(seed)
    ss = 4
    S = size * ss
    m = S * 0.18
    gx = np.linspace(m, S - m, 3)
    gy = np.linspace(m, S - m, 4)
    out = []
    for _ in range(n):
        c = np.zeros((S, S), np.uint8)
        th = int(S * 0.075)
        for _k in range(rng.integers(2, 5)):
            i, j = rng.integers(0, 3), rng.integers(0, 4)
            di, dj = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1, 1])
            i2, j2 = int(np.clip(i + di, 0, 2)), int(np.clip(j + dj, 0, 3))
            p1 = (int(gx[i]), int(gy[j]))
            p2 = (int(gx[i2]), int(gy[j2]))
            if rng.random() < 0.25:
                r = int(S * rng.uniform(0.12, 0.22))
                a0 = int(rng.integers(0, 4)) * 90
                cv2.ellipse(c, p1, (r, r), 0, a0, a0 + 180, 255, th, cv2.LINE_AA)
            elif p1 != p2:
                cv2.line(c, p1, p2, 255, th, cv2.LINE_AA)
        if rng.random() < 0.4:
            cv2.circle(c, (int(gx[rng.integers(0, 3)]), int(gy[0] - m * 0.3)), th, 255, -1, cv2.LINE_AA)
        out.append(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    return out


@lru_cache(maxsize=32)
def _font(path, size):
    return ImageFont.truetype(path, size)


def text_mask(s, size, path=FONT_SERIF, pad=8, spacing=0):
    f = _font(path, size)
    l, t, r, b = f.getbbox(s)
    if spacing:
        widths = [f.getbbox(ch)[2] for ch in s]
        tw = sum(widths) + spacing * (len(s) - 1)
    else:
        tw = r - l
    im = Image.new("L", (tw + 2 * pad, b - t + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    if spacing:
        x = pad
        for ch, wd in zip(s, widths):
            d.text((x, pad - t), ch, fill=255, font=f)
            x += wd + spacing
    else:
        d.text((pad - l, pad - t), s, fill=255, font=f)
    return np.asarray(im, np.float32) / 255


def paste_mask(dst_mask, m, x, y):
    """Add mask m into dst_mask with top-left at (x, y), clipped."""
    h, w = m.shape
    H_, W_ = dst_mask.shape
    x0, y0 = int(round(x)), int(round(y))
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W_, x0 + w), min(H_, y0 + h)
    if xa >= xb or ya >= yb:
        return
    dst_mask[ya:yb, xa:xb] = np.maximum(dst_mask[ya:yb, xa:xb], m[ya - y0:yb - y0, xa - x0:xb - x0])


def rounded_rect_mask(h, w, cx, cy, hw, hh, r, soft=1.0):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(xs - cx) - (hw - r)
    qy = np.abs(ys - cy) - (hh - r)
    d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
    return np.clip(0.5 - d / soft, 0, 1).astype(np.float32), d


class Patches:
    """Slowly drifting patchy mask so caustics/light come and go instead of tiling the frame."""

    def __init__(self, w=W, h=H, cell=260, seed=0, lo=0.42, hi=0.68):
        self.w, self.h = w, h
        self.n = value_noise(int(h * 1.6), int(w * 1.6), cell, seed=seed, octaves=3)
        self.lo, self.hi = lo, hi

    def __call__(self, t, vx=9.0, vy=4.0):
        ox = int(abs(t * vx)) % (self.n.shape[1] - self.w)
        oy = int(abs(t * vy)) % (self.n.shape[0] - self.h)
        m = self.n[oy:oy + self.h, ox:ox + self.w]
        return np.clip((m - self.lo) / (self.hi - self.lo), 0, 1) ** 1.5


# ------------------------------------------------------------------ particles
class Snow:
    """Marine snow: drifting specks at two depths."""

    def __init__(self, n=260, seed=0, w=W, h=H, vy=(-10, 6)):
        r = np.random.default_rng(seed)
        self.w, self.h = w, h
        self.x0 = r.random(n) * w
        self.y0 = r.random(n) * h
        self.vy = r.uniform(*vy, n)
        self.vx = r.uniform(-4, 4, n)
        self.ph = r.random(n) * 6.283
        self.sz = r.uniform(0.6, 2.0, n) ** 1.6
        self.br = r.uniform(0.25, 1.0, n)
        self.near = r.random(n) < 0.12

    def mask(self, t, ox=0.0, oy=0.0):
        far = np.zeros((self.h, self.w), np.float32)
        near = np.zeros_like(far)
        x = (self.x0 + self.vx * t + 7 * np.sin(t * 0.4 + self.ph) + ox) % self.w
        y = (self.y0 + self.vy * t + oy) % self.h
        for i in range(len(x)):
            tgt = near if self.near[i] else far
            rad = self.sz[i] * (4.0 if self.near[i] else 1.0)
            cv2.circle(tgt, (int(x[i] * 16), int(y[i] * 16)), max(1, int(rad * 16)), float(self.br[i]), -1, cv2.LINE_AA, shift=4)
        return far + blur(near, 3.0) * 0.6


def bubble_sprite(r, ss=4):
    S = int(2 * r * ss + 8)
    c = np.zeros((S, S), np.float32)
    ctr = (S // 2, S // 2)
    R = int(r * ss)
    cv2.circle(c, ctr, R, 0.10, -1, cv2.LINE_AA)
    cv2.circle(c, ctr, R, 0.85, max(1, int(ss * 1.2)), cv2.LINE_AA)
    cv2.ellipse(c, (ctr[0] - R // 3, ctr[1] - R // 3), (R // 4, R // 6), -35, 0, 360, 1.0, -1, cv2.LINE_AA)
    return cv2.resize(c, (S // ss, S // ss), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ finishing / output
_VIG = None
_GRAIN = None


def finish(img, idx, vignette=0.45, grain=0.022, bloom_amt=0.55, bloom_th=0.62, ca=1.2):
    global _VIG, _GRAIN
    if _VIG is None:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        _VIG = (d ** 2.2)[..., None]
        r = np.random.default_rng(99)
        _GRAIN = [r.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(8)]
    hi = np.clip(img - bloom_th, 0, None)
    img = img + (blur(hi, 10) * 0.5 + blur(hi, 40) * 0.5) * bloom_amt
    img = img * (1 - vignette * _VIG)
    if ca:
        for ch, s in ((0, 1 + ca * 0.0007), (2, 1 - ca * 0.0007)):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
            img[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    g = cv2.resize(_GRAIN[idx % 8], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    img = img + g * grain * (0.35 + 0.65 * (1 - np.clip(img.mean(-1, keepdims=True), 0, 1)))
    return np.clip(img, 0, 1)


class Writer:
    def __init__(self, path, fps=FPS, size=(W, H), crf=17):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def save_png(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
