"""v2 'print' toolkit: code-separated ink plates of the character, paper grounds, marks, type roles.

Design rules (see 07_设计系统_v2.md):
  - her = ultramarine / ice / ink-navy (three flat inks, no gradients); you (the human) = orange; grey = absence
  - light is printed: flat steps and dot screens, never soft gradients
  - depth comes from a second ink or a plate printed out of register
  - type roles: serif = her words, sans = interface / the user, mono = system
"""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lib import CHAR, H, SRC, W, affine, blur, hexc, value_noise

INK = dict(
    paper=hexc("#0b1426"),     # dark navy paper (never pure black)
    paper_hi=hexc("#13213d"),  # printed light, step 1
    paper_hi2=hexc("#1b2e55"), # printed light, step 2
    navy=hexc("#050913"),      # her dark ink
    blue=hexc("#3156e0"),      # her ultramarine
    blue_dim=hexc("#1f3a9c"),
    ice=hexc("#e6edf5"),       # her light ink
    orange=hexc("#ff6a1f"),    # you
    orange_dim=hexc("#a8441a"),
    cream=hexc("#ece6d9"),     # the interface's paper (the screen world)
    cream_ink=hexc("#151a26"),
    grey=hexc("#7d8597"),
)
FONT = dict(serif="C:/Windows/Fonts/NotoSerifSC-VF.ttf", sans="C:/Windows/Fonts/NotoSansSC-VF.ttf",
            mono="C:/Windows/Fonts/CascadiaMono.ttf")


# ------------------------------------------------------------------ ink separation
def _clean(m, min_area, close=5, smooth=2.0):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(m.shape, np.uint8)
    big = np.nonzero(st[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
    keep[np.isin(lab, big)] = 1
    return (cv2.GaussianBlur(keep.astype(np.float32), (0, 0), smooth) > 0.5).astype(np.float32)


class Inks:
    """deepseekgirl.png separated into three flat plates (dark / mid / light) at `scale` of the source."""

    def __init__(self, scale=0.5, t1=0.36, t2=0.72, faceless=True, path=None, min_k=1.0):
        im = np.asarray(Image.open(path or (SRC / "deepseekgirl.png")).convert("RGBA")).astype(np.float32) / 255
        im[..., 3] = np.clip(im[..., 3] * 1.01, 0, 1)
        pm = im.copy()
        pm[..., :3] *= pm[..., 3:4]
        pm = cv2.resize(pm, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        a = pm[..., 3]
        rgb = pm[..., :3] / np.maximum(a[..., None], 1e-3)
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        lum = cv2.bilateralFilter(lum, 9, 0.08, 6)
        k = scale / 0.5 * min_k
        fg = _clean(a > 0.5, int(400 * k * k), close=3, smooth=1.5)
        # warm hues (skin, the inside of the mouth, blush) print in the light ink unless they are line-dark,
        # so a face keeps only its drawn lines: an open smile reads as its outline, not as a mid-tone blob
        rb = rgb[..., 0] - rgb[..., 2]
        mouth = (rb > 0.15) & (lum > 0.30) & (lum < 0.84)        # saturated pink-red: an open mouth prints solid dark
        warm = (rb > 0.02) & (lum >= 0.84)                       # skin, blush: light
        dark = _clean(((lum < t1) | mouth) * fg, int(70 * k * k)) * fg
        light = _clean(((lum > t2) | warm) * fg, int(70 * k * k)) * fg
        light = light * (1 - dark)
        if faceless:
            # the face becomes one flat light shape: drop every small dark/mid island inside it
            ys, xs = np.mgrid[0:fg.shape[0], 0:fg.shape[1]].astype(np.float32)
            fx, fy = CHAR["face"][0] * scale, (CHAR["face"][1] + 30) * scale
            ell = (((xs - fx) / (300 * scale)) ** 2 + ((ys - fy) / (250 * scale)) ** 2) < 1
            for m in (dark,):
                n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
                for i in range(1, n):
                    if st[i, cv2.CC_STAT_AREA] < int(3000 * k * k):
                        comp = lab == i
                        if ell[comp].mean() > 0.5:
                            m[comp] = 0
            mid0 = np.clip(fg - dark - light, 0, 1)
            n, lab, st, _ = cv2.connectedComponentsWithStats((mid0 > 0.5).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                comp = lab == i
                if st[i, cv2.CC_STAT_AREA] < int(2500 * k * k) and ell[comp].mean() > 0.6:
                    light[comp] = 1
        mid = np.clip(fg - dark - light, 0, 1)
        self.scale = scale
        self.plates = np.stack([dark, mid, light, fg], axis=2).astype(np.float32)

    def warp(self, anchor, pos, sc, rot=0.0, size=(W, H)):
        """Plates in frame space. anchor/pos/sc in source-pixel terms (like Sprite)."""
        M = affine((anchor[0] * self.scale, anchor[1] * self.scale), pos, sc / self.scale, rot)
        return cv2.warpAffine(self.plates, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def print_figure(img, P, inks=("navy", "blue", "ice"), shadow=None, shadow_off=(14, 10), keyline="ice", key_w=3,
                 key_amt=0.0, alpha=1.0):
    """Print warped plates P (H, W, 4: dark, mid, light, fg) onto img in flat inks.
    shadow: ink name for an out-of-register copy of the whole figure under it."""
    dark, mid, light, fg = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    if shadow:
        sh = nudge(fg, shadow_off[0], shadow_off[1])
        img = fill(img, sh * alpha, INK[shadow])
    if key_amt:
        k = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * key_w + 1, 2 * key_w + 1)))
        img = fill(img, np.clip(k - fg, 0, 1) * key_amt * alpha, INK[keyline])
    for m, ink in ((mid, inks[1]), (light, inks[2]), (dark, inks[0])):
        img = fill(img, m * alpha, INK[ink] if isinstance(ink, str) else ink)
    return img


def nudge(m, dx, dy):
    """Shift without wrap-around (np.roll would bleed the opposite edge back in)."""
    return cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (m.shape[1], m.shape[0]),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fill(img, m, color):
    return img * (1 - m[..., None]) + m[..., None] * np.asarray(color, np.float32)


# ------------------------------------------------------------------ grounds
_G = {}


def _screen(h, w, spacing=7.0, angle=45):
    key = (h, w, spacing, angle)
    if key not in _G:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        u, v = xs * c + ys * s, -xs * s + ys * c
        du = (u / spacing) % 1 - 0.5
        dv = (v / spacing) % 1 - 0.5
        _G[key] = np.sqrt(du * du + dv * dv) * spacing  # distance to nearest dot centre, px
    return _G[key]


def dots(h, w, density, spacing=7.0, angle=45):
    """Halftone dot screen: density (h, w) in [0, 1] -> coverage mask with crisp round dots."""
    d = _screen(h, w, spacing, angle)
    r = np.sqrt(np.clip(density, 0, 1)) * spacing * 0.56
    return np.clip(r - d + 0.5, 0, 1)


def grain(h, w, seed=0):
    key = ("grain", h, w, seed)
    if key not in _G:
        r = np.random.default_rng(seed)
        g = r.normal(0, 1, (h, w)).astype(np.float32)
        g = g * 0.6 + cv2.GaussianBlur(g, (0, 0), 1.2) * 0.8
        fib = value_noise(h, w, 90, seed + 7, 3) - 0.5
        _G[key] = g * 0.010 + fib * 0.025
    return _G[key]


def pool_steps(h, w, cx, cy, rx, ry, steps=2, rough=0.04, seed=0):
    """Printed light: nested flat ellipses with slightly rough edges. Returns list of masks, outer first."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    n = (value_noise(h, w, 60, seed, 2) - 0.5) * rough * 2
    out = []
    for k in range(steps):
        lim = 1.0 - k * (0.42 / max(1, steps - 1)) if steps > 1 else 1.0
        out.append(np.clip((lim + n - d) * 180, 0, 1).astype(np.float32))
    return out


def ground(pools=(), base="paper", screen=0.0, seed=0):
    """Dark navy paper: flat tone + fibres/grain + optional printed light pools (flat steps) + dot screen."""
    img = np.broadcast_to(INK[base], (H, W, 3)).astype(np.float32).copy()
    lit = np.zeros((H, W), np.float32)
    for (cx, cy, rx, ry, steps) in pools:
        ms = pool_steps(H, W, cx, cy, rx, ry, steps, seed=seed + int(cx))
        for k, m in enumerate(ms):
            img = fill(img, m, INK["paper_hi"] if k == 0 else INK["paper_hi2"])
            lit = np.maximum(lit, m * (k + 1) / steps)
    if screen:
        img = fill(img, dots(H, W, lit * screen + 0.06) * 0.35, INK["paper_hi2"])
    img += grain(H, W, seed)[..., None]
    return img


# ------------------------------------------------------------------ marks and type
def crop_marks(img, color="ice", amt=0.55, inset=36, length=26, gap=8):
    m = np.zeros((H, W), np.float32)
    for (x, y, sx, sy) in ((inset, inset, 1, 1), (W - inset, inset, -1, 1), (inset, H - inset, 1, -1), (W - inset, H - inset, -1, -1)):
        cv2.line(m, (x - sx * (gap + length), y), (x - sx * gap, y), 1.0, 1, cv2.LINE_AA)
        cv2.line(m, (x, y - sy * (gap + length)), (x, y - sy * gap), 1.0, 1, cv2.LINE_AA)
    cx = W // 2
    cv2.circle(m, (cx, H - inset + 4), 7, 1.0, 1, cv2.LINE_AA)
    cv2.line(m, (cx - 12, H - inset + 4), (cx + 12, H - inset + 4), 1.0, 1, cv2.LINE_AA)
    cv2.line(m, (cx, H - inset - 8), (cx, H - inset + 16), 1.0, 1, cv2.LINE_AA)
    return fill(img, m * amt, INK[color])


@lru_cache(maxsize=64)
def _font(role, size):
    f = ImageFont.truetype(FONT[role], size)
    if role in ("serif", "sans"):
        try:
            f.set_variation_by_axes([600 if role == "serif" else 450])
        except Exception:
            pass
    return f


def text(s, size, role="mono", tracking=0):
    """Rasterise s -> float mask (h, w)."""
    f = _font(role, size)
    l, t, r, b = f.getbbox(s)
    if tracking:
        ws = [f.getlength(ch) for ch in s]
        w = int(sum(ws) + tracking * (len(s) - 1)) + 4
    else:
        w = int(r - l) + 4
    im = Image.new("L", (w, int(b - t) + 6), 0)
    d = ImageDraw.Draw(im)
    if tracking:
        x = 2
        for ch, cw in zip(s, ws):
            d.text((x, 3 - t), ch, fill=255, font=f)
            x += cw + tracking
    else:
        d.text((2 - l, 3 - t), s, fill=255, font=f)
    return np.asarray(im, np.float32) / 255


def put(img, mask, x, y, color, amt=1.0, anchor="lt"):
    h, w = mask.shape
    if anchor[0] == "c":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "c":
        y -= h / 2
    elif anchor[1] == "b":
        y -= h
    x0, y0 = int(round(x)), int(round(y))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return img
    m = mask[ya - y0:yb - y0, xa - x0:xb - x0] * amt
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - m[..., None]) + m[..., None] * np.asarray(INK[color] if isinstance(color, str) else color, np.float32)
    return img


def caption_band(img, line_label, y=930, color="ice"):
    """The fixed band where a sung line sits (placeholder label in style frames; real text comes from the user's lyric file)."""
    band = np.zeros((H, W), np.float32)
    cv2.rectangle(band, (0, y - 44), (W, y + 44), 1.0, -1)
    img = fill(img, band * 0.55, INK["navy"])
    if not line_label:
        return img
    m = text(line_label, 40, "serif", tracking=6)
    return put(img, m, W / 2, y, color, 0.95, "cc")


# ------------------------------------------------------------------ her emblem: the user's whale avatar, traced
class Emblem:
    """assets/from_user/ui_ref/avatar_whale.png -> two smooth masks (body, inner white), renderable at any size."""

    def __init__(self, path=None, up=8):
        from lib import ROOT
        path = path or (ROOT / "assets" / "from_user" / "ui_ref" / "avatar_whale.png")
        rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
        blue = ((rgb[..., 2] - rgb[..., 0]) > 0.18).astype(np.uint8)
        white = (rgb.min(axis=2) > 0.80).astype(np.uint8)
        # white that is not connected to the border = belly / eye
        n, lab = cv2.connectedComponents(white, connectivity=4)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        inner = np.isin(lab, [i for i in range(1, n) if i not in border]).astype(np.float32)
        body = np.clip(blue + inner, 0, 1).astype(np.float32)
        ys, xs = np.nonzero(body)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = 4
        body, inner = body[y0 - pad:y1 + pad, x0 - pad:x1 + pad], inner[y0 - pad:y1 + pad, x0 - pad:x1 + pad]

        def contours(m, sigma=2.2):
            m = cv2.GaussianBlur(m, (0, 0), 0.7)
            cs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            out = []
            for c in cs:
                c = c[:, 0, :].astype(np.float32)
                if len(c) < 12:
                    continue
                k = int(sigma * 3)
                pad_ = np.concatenate([c[-k:], c, c[:k]])
                kern = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
                kern /= kern.sum()
                sm = np.stack([np.convolve(pad_[:, j], kern, mode="same")[k:-k] for j in (0, 1)], 1)
                out.append(sm)
            return out
        self.h0, self.w0 = body.shape
        self.c_body, self.c_inner = contours(body), contours(inner, 1.6)
        self.body = self._draw(self.c_body, 400)

    def _draw(self, cs, height):
        s = height / self.h0
        w = max(1, int(round(self.w0 * s)))
        m = np.zeros((int(height), w), np.uint8)
        polys = [np.round(c * s * 16).astype(np.int32) for c in cs]
        cv2.fillPoly(m, polys, 255, cv2.LINE_AA, shift=4)
        return m.astype(np.float32) / 255

    def masks(self, height):
        """(body, inner) masks scaled so the emblem is `height` px tall (drawn from smoothed contours: crisp at any size)."""
        return self._draw(self.c_body, height), self._draw(self.c_inner, height)

    def stamp(self, img, cx, cy, height, body_col, inner_col, alpha=1.0):
        b, i = self.masks(height)
        h_, w_ = b.shape
        img = put(img, b, cx, cy, body_col, alpha, "cc")
        return put(img, i, cx, cy, inner_col, alpha, "cc")


# ------------------------------------------------------------------ shapes
def whale_mask(length, width_scale=1.0):
    """Top-down whale silhouette (as a shadow on the seabed), head pointing left."""
    xs = np.linspace(0, 0.82, 48)
    u = xs / 0.82
    hw = (0.12 * u ** 0.45 * (1 - u) ** 1.1 + 0.012 * u) * width_scale
    upper = [(x, -h) for x, h in zip(xs, hw)]
    fin_i = 15
    fin = [(0.30, -hw[fin_i] - 0.02), (0.44, -0.30 * width_scale), (0.40, -0.27 * width_scale), (0.34, -hw[fin_i + 3])]
    upper = upper[:fin_i] + fin + upper[fin_i + 3:]
    fluke = [(0.88, -0.09), (0.99, -0.14), (0.97, -0.07), (0.93, 0.0), (0.97, 0.07), (0.99, 0.14), (0.88, 0.09)]
    lower = [(x, -y) for x, y in reversed(upper)]
    pts = upper + fluke + lower
    pad = int(length * 0.35)
    m = np.zeros((pad * 2, int(length * 1.05)), np.uint8)
    poly = np.array([[x * length, pad + y * length] for x, y in pts], np.int32)
    cv2.fillPoly(m, [poly], 255, cv2.LINE_AA)
    return m.astype(np.float32) / 255
