"""Cyber 'construction' animations: nothing simply appears.

  decode()        text: scrambled glyphs resolve left to right
  box_build()     a frame: corner ticks -> edges trace out -> a scanline sweeps the fill in -> returns content alpha
  scan_reveal()   reveal a finished layer behind a bright scanline with a ragged block front
  assemble()      particles fly in and settle onto a mask; the solid fades in under them
  print_run()     a printed figure: plates arrive one by one, sliding into register
  mosaic_in()     a patch resolves from big pixels to sharp
"""
import math

import cv2
import numpy as np

from cyber import C, glass, rrect
from lib import H, W, ease_out, seg, smooth

SCRAMBLE = "░▒▓█<>/\\|#%&@$*+=-_0123456789ABCDEF"      # system text only (HUD, tags, code)
SAME = "SAME"                                         # ascii_pool=SAME: a letter becomes another letter of its case


# ------------------------------------------------------------------ look-alike glyphs
# The wrong characters a decode shows on its way to the right one are characters that LOOK like it
# (nearest neighbours by rendered shape among the 3755 most common hanzi), never words chosen for their meaning.
_LOOK = {}
_POOL = {}


def _glyph_feat(ch, size=32, cell=40, out=20):
    from PIL import Image, ImageDraw, ImageFont
    if "font" not in _POOL:
        _POOL["font"] = ImageFont.truetype("C:/Windows/Fonts/NotoSansSC-VF.ttf", size)
    im = Image.new("L", (cell, cell), 0)
    ImageDraw.Draw(im).text((cell / 2, cell / 2), ch, fill=255, font=_POOL["font"], anchor="mm")
    a = cv2.GaussianBlur(np.asarray(im, np.float32) / 255, (0, 0), 1.1)
    a = cv2.resize(a, (out, out), interpolation=cv2.INTER_AREA).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def _pool():
    if "chars" not in _POOL:
        from lib import ROOT
        cache = ROOT / "work" / "cache" / "glyph_pool.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            _POOL["chars"], _POOL["feats"] = list(str(z["chars"])), z["feats"]
        else:
            chars = []
            for hi in range(0xB0, 0xD8):                       # GB2312 level 1: the 3755 most common hanzi
                for lo in range(0xA1, 0xFF):
                    try:
                        chars.append(bytes([hi, lo]).decode("gb2312"))
                    except UnicodeDecodeError:
                        pass
            feats = np.stack([_glyph_feat(c) for c in chars]).astype(np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, chars="".join(chars), feats=feats)
            _POOL["chars"], _POOL["feats"] = chars, feats
    return _POOL["chars"], _POOL["feats"]


def lookalikes(ch, k=6):
    """The k common hanzi whose rendered shape is closest to ch (ch itself excluded)."""
    if ch not in _LOOK:
        chars, feats = _pool()
        sims = feats @ _glyph_feat(ch)
        order = np.argsort(-sims)
        _LOOK[ch] = [chars[i] for i in order[:k + 1] if chars[i] != ch][:k]
    return _LOOK[ch]


def _is_hanzi(ch):
    return 0x3400 <= ord(ch) <= 0x9FFF


def wrong_char(ch, r, ascii_pool=None):
    """One plausible wrong glyph for ch at this instant."""
    if _is_hanzi(ch):
        la = lookalikes(ch)
        return la[r.integers(0, len(la))] if la else ch
    if ch == " " or ord(ch) >= 0x2E80:                     # spaces and full-width punctuation stay put
        return ch
    if ascii_pool == SAME:
        if ch.isdigit():
            return str(r.integers(0, 10))
        if ch.isalpha():
            base = "abcdefghijklmnopqrstuvwxyz"
            c = base[r.integers(0, 26)]
            return c.upper() if ch.isupper() else c
        return ch
    pool = ascii_pool or SCRAMBLE
    return pool[r.integers(0, len(pool))]


def _rng(t, salt=0):
    return np.random.default_rng((int(t * 30) * 7919 + salt) & 0x7FFFFFFF)


def decode(s, t, t0, dur=0.35, head=3, salt=0, ascii_pool=None):
    """String as it reads at time t: resolved prefix + a few look-alike glyphs at the head; nothing beyond."""
    if t < t0:
        return ""
    k = smooth(seg(t, t0, t0 + dur))
    n = len(s)
    done = int(n * k)
    if done >= n:
        return s
    r = _rng(t, salt + n)
    out = list(s[:done])
    for i in range(done, min(n, done + head)):
        out.append(wrong_char(s[i], r, ascii_pool))
    return "".join(out)


def box_build(fr, t, t0, rect, r=14, dur=0.45, fill=None, fill_alpha=0.85, line=None, line_amt=1.0, glow=0.6, out=None):
    """Draw a box as it assembles. Returns the content alpha (0 until the fill is in, then -> 1).
    out=(t1, dur1) runs it backwards (de-construction)."""
    if t < t0:
        return 0.0
    p = seg(t, t0, t0 + dur)
    if out is not None and t >= out[0]:
        p = 1 - seg(t, out[0], out[0] + out[1])
        if p <= 0:
            return 0.0
    x0, y0, x1, y1 = [float(v) for v in rect]
    fill = C["glass"] if fill is None else fill
    line = C["glass_line"] if line is None else line
    w_, h_ = x1 - x0, y1 - y0
    # 1. corner ticks snap in
    pt = smooth(seg(p, 0.0, 0.18))
    L = min(18, w_ / 3, h_ / 3)
    m = np.zeros((H, W), np.float32)
    if pt > 0:
        for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(m, (int(x), int(y)), (int(x + sx * L * pt), int(y)), 1.0, 2, cv2.LINE_AA)
            cv2.line(m, (int(x), int(y)), (int(x), int(y + sy * L * pt)), 1.0, 2, cv2.LINE_AA)
    # 2. edges trace from the corners toward the middle of each side
    pe = ease_out(seg(p, 0.12, 0.5), 2)
    if pe > 0:
        hx, hy = w_ / 2 * pe, h_ / 2 * pe
        for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(m, (int(x), int(y)), (int(x + sx * hx), int(y)), 1.0, 1, cv2.LINE_AA)
            cv2.line(m, (int(x), int(y)), (int(x), int(y + sy * hy)), 1.0, 1, cv2.LINE_AA)
    # 3. the fill sweeps down behind a bright scanline
    pf = seg(p, 0.35, 0.8)
    if pf > 0:
        yy = y0 + h_ * ease_out(pf, 2)
        body = rrect(np.zeros((H, W), np.float32), x0, y0, x1, y1, r)
        body[int(min(H, max(0, yy))):] = 0
        fr.fill(body, fill, fill_alpha)
        if pf < 1:
            sl = np.zeros((H, W), np.float32)
            cv2.line(sl, (int(x0 + 2), int(yy)), (int(x1 - 2), int(yy)), 1.0, 2)
            r_ = _rng(t, int(x0))
            for _ in range(6):
                bx = r_.uniform(x0, x1 - 30)
                cv2.rectangle(sl, (int(bx), int(yy - r_.uniform(2, 10))), (int(bx + r_.uniform(8, 40)), int(yy)), 0.6, -1)
            fr.fill(sl, C["cyan"], 0.9, glow=0.8)
    if p >= 0.5:
        outline = rrect(np.zeros((H, W), np.float32), x0, y0, x1, y1, r)
        inner = rrect(np.zeros((H, W), np.float32), x0 + 1.3, y0 + 1.3, x1 - 1.3, y1 - 1.3, max(0, r - 1.3))
        m = np.maximum(m * (1 - smooth(seg(p, 0.5, 0.7))), np.clip(outline - inner, 0, 1))
    fr.fill(m, line, line_amt, glow=glow)
    return smooth(seg(p, 0.7, 1.0))


def scan_reveal(base, layer, t, t0, dur, rect, down=True, edge=True, fr=None):
    """Composite `layer` over `base` inside rect, revealed by a scanline moving down (or up). Returns the image."""
    x0, y0, x1, y1 = [int(v) for v in rect]
    if t < t0:
        return base
    p = ease_out(seg(t, t0, t0 + dur), 2)
    hh = y1 - y0
    yy = y0 + int(hh * p) if down else y1 - int(hh * p)
    out = base.copy()
    if down:
        out[y0:yy, x0:x1] = layer[y0:yy, x0:x1]
    else:
        out[yy:y1, x0:x1] = layer[yy:y1, x0:x1]
    if edge and p < 1:
        r_ = _rng(t, x0)
        band = np.zeros((H, W), np.float32)
        for _ in range(14):
            bx = r_.uniform(x0, x1 - 40)
            bh = r_.uniform(3, 18)
            ya, yb = (yy, yy + bh) if down else (yy - bh, yy)
            cv2.rectangle(band, (int(bx), int(ya)), (int(bx + r_.uniform(10, 90)), int(yb)), 1.0, -1)
        band[:, :x0] = 0
        band[:, x1:] = 0
        a = band[..., None] * 0.85
        out = out * (1 - a) + a * layer
        line = np.zeros((H, W), np.float32)
        cv2.line(line, (x0, yy), (x1, yy), 1.0, 2)
        out = out * (1 - line[..., None]) + line[..., None] * C["cyan"]
        if fr is not None:
            fr.glow += (line[..., None] * 0.9 + band[..., None] * 0.15) * C["cyan"]
    return out


_ASM = {}


def assemble(fr, key, mask, t, t0, dur=0.6, color=None, n=900, src=None, spread=420, seed=0, glow=1.0, size=1.6):
    """Particles sampled from `mask` fly in from `src` (point) or a ring around the shape, landing staggered.
    Returns the alpha the solid shape should be drawn with (0 -> 1 as the particles land)."""
    if t < t0:
        return 0.0
    color = C["cyan"] if color is None else color
    if key not in _ASM:
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) == 0:
            _ASM[key] = None
        else:
            r = np.random.default_rng(seed)
            idx = r.choice(len(xs), size=min(n, len(xs)), replace=False)
            tx, ty = xs[idx].astype(np.float32), ys[idx].astype(np.float32)
            if src is None:
                ang = r.uniform(0, 2 * math.pi, len(tx))
                rad = r.uniform(0.4, 1.0, len(tx)) * spread
                sx, sy = tx + np.cos(ang) * rad, ty + np.sin(ang) * rad
            else:
                sx = np.full(len(tx), src[0], np.float32) + r.normal(0, 18, len(tx))
                sy = np.full(len(tx), src[1], np.float32) + r.normal(0, 18, len(tx))
            delay = r.uniform(0, 0.45, len(tx))
            curl = r.normal(0, 1, len(tx)) * 60
            _ASM[key] = (tx, ty, sx, sy, delay, curl)
    data = _ASM[key]
    if data is None:
        return 1.0
    tx, ty, sx, sy, delay, curl = data
    p = (t - t0) / dur
    if p >= 1.15:
        return 1.0
    k = np.clip((p - delay) / 0.55, 0, 1)
    e = 1 - (1 - k) ** 3
    nx, ny = -(ty - sy), (tx - sx)
    nl = np.sqrt(nx * nx + ny * ny) + 1e-6
    bow = np.sin(e * math.pi) * curl
    px = sx + (tx - sx) * e + nx / nl * bow
    py = sy + (ty - sy) * e + ny / nl * bow
    vis = (k > 0) & (k < 1)
    landed = k >= 1
    buf = np.zeros((H, W), np.float32)
    for sel, val in ((vis, 1.0), (landed, 0.6 * (1 - smooth(seg(p, 0.9, 1.15))))):
        xi, yi = px[sel].astype(np.int32), py[sel].astype(np.int32)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        np.add.at(buf, (yi[ok], xi[ok]), val)
    # short trails
    k2 = np.clip((p - 0.04 - delay) / 0.55, 0, 1)
    e2 = 1 - (1 - k2) ** 3
    bow2 = np.sin(e2 * math.pi) * curl
    qx = sx + (tx - sx) * e2 + nx / nl * bow2
    qy = sy + (ty - sy) * e2 + ny / nl * bow2
    sel = vis & (k2 > 0)
    xi, yi = qx[sel].astype(np.int32), qy[sel].astype(np.int32)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    np.add.at(buf, (yi[ok], xi[ok]), 0.45)
    buf = np.clip(cv2.GaussianBlur(buf, (0, 0), size * 0.6) * 2.2, 0, 1)
    fr.add(buf, color, 1.0, glow=glow)
    return float(np.clip((p - 0.55) / 0.5, 0, 1))


def print_run(fr, P, t, t0, dur=0.5, inks=None, misreg=30, shadow=True, rim=True):
    """Print the plates of a figure one after another, each sliding into register (a press run)."""
    from cyber import print_figure
    if t < t0:
        return fr
    inks = inks or (C["ink"], C["blue"], C["ice"])
    order = [(1, inks[1], 0.0), (2, inks[2], 0.25), (0, inks[0], 0.5)]
    dark, mid, light, fg = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    p = seg(t, t0, t0 + dur)
    if p >= 1:
        return print_figure(fr, P, inks=inks, rim=rim)
    if shadow:
        a = smooth(seg(p, 0.0, 0.3))
        sh = cv2.warpAffine(fg, np.float32([[1, 0, 10], [0, 1, 7]]), (W, H))
        fr.fill(sh * a, C["blue_dim"], 0.9)
    for (ch, ink, st) in order:
        q = seg(p, st, st + 0.5)
        if q <= 0:
            continue
        off = (1 - ease_out(q, 3)) * misreg
        M = np.float32([[1, 0, off * (1 if ch != 2 else -1)], [0, 1, -off * 0.6]])
        m = cv2.warpAffine(P[..., ch], M, (W, H))
        fr.fill(m * smooth(seg(q, 0, 0.3)), ink)
    if rim and p > 0.8:
        e = np.clip(fg - cv2.warpAffine(fg, np.float32([[1, 0, 0], [0, 1, 4]]), (W, H), borderMode=cv2.BORDER_REPLICATE), 0, 1)
        fr.fill(cv2.GaussianBlur(e, (0, 0), 0.8) * smooth(seg(p, 0.8, 1.0)), C["cyan"], 0.9, glow=0.8)
    return fr


def mosaic_in(img, t, t0, dur=0.45, levels=(48, 24, 12, 6, 3, 1)):
    """A patch (h, w, C) seen through decreasing pixel size."""
    if t < t0:
        return None
    k = seg(t, t0, t0 + dur)
    i = min(len(levels) - 1, int(k * len(levels)))
    s = levels[i]
    if s <= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


_PIX = {}


def pixel_build(fr, t, t0, rect, block=16, dur=0.75, r=12, fill=None, fill_alpha=0.93, line=None, seed=0):
    """A panel assembled from pixel blocks: the perimeter blocks land first (traced clockwise from the top-left),
    then the interior rains in top-down with noise. Each block pops in from its centre and flashes cyan;
    blocks about to land flicker as faint pre-echoes. At the end the gaps close into one seamless glass panel.
    Returns the content alpha (0 until the panel is whole)."""
    if t < t0:
        return 0.0
    x0, y0, x1, y1 = [int(v) for v in rect]
    w, h = x1 - x0, y1 - y0
    fill = C["glass"] if fill is None else fill
    line = C["glass_line"] if line is None else line
    end = t0 + dur
    key = (x0, y0, x1, y1, block, seed, dur)
    if key not in _PIX:
        bxn, byn = -(-w // block), -(-h // block)
        rng = np.random.default_rng(seed)
        jj, ii = np.meshgrid(np.arange(bxn), np.arange(byn))
        per = (ii == 0) | (jj == 0) | (ii == byn - 1) | (jj == bxn - 1)
        # clockwise parameter around the perimeter, starting at the top-left
        s = np.where(ii == 0, jj / bxn * 0.25,
            np.where(jj == bxn - 1, 0.25 + ii / byn * 0.25,
            np.where(ii == byn - 1, 0.5 + (bxn - 1 - jj) / bxn * 0.25, 0.75 + (byn - 1 - ii) / byn * 0.25)))
        arr = np.where(per, 0.30 * s + rng.uniform(0, 0.03, per.shape),
                       0.22 + 0.55 * (0.72 * ii / byn + 0.28 * rng.random(per.shape)))
        arr = (arr * dur).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        bi, bj = yy // block, xx // block
        ly = (yy % block) - (block - 1) / 2
        lx = (xx % block) - (block - 1) / 2
        cheb = np.maximum(np.abs(lx), np.abs(ly)).astype(np.float32)
        body = rrect(np.zeros((H, W), np.float32), x0, y0, x1, y1, r)[y0:y1, x0:x1]
        inner = rrect(np.zeros((H, W), np.float32), x0 + 1.3, y0 + 1.3, x1 - 1.3, y1 - 1.3, max(0, r - 1.3))[y0:y1, x0:x1]
        flick = rng.random((64,) + per.shape) < 0.3
        _PIX[key] = (arr, bi, bj, cheb, body, np.clip(body - inner, 0, 1), flick)
    arr, bi, bj, cheb, body, edge, flick = _PIX[key]
    if t >= end:
        glass(fr, x0, y0, x1, y1, r=r, alpha=fill_alpha, line=line)
        return 1.0
    age = (t - t0) - arr
    k = np.clip(age / 0.07, 0, 1)
    k = 1 - (1 - k) ** 3
    fl = np.where(age >= 0, np.exp(-np.maximum(age, 0) * 16), 0).astype(np.float32)
    pre = ((age > -0.12) & (age < 0) & flick[int(t * 30) % 64]).astype(np.float32)
    close = smooth(seg(t, end - 0.14, end))
    gap = 1.2 * (1 - close)
    K, FL, PRE = k[bi, bj], fl[bi, bj], pre[bi, bj]
    half = K * (cheb.max() + 0.5) - gap
    inside = (cheb <= half).astype(np.float32) * body
    ghost = (cheb <= (cheb.max() - 1.0)).astype(np.float32) * PRE * body
    img = fr.img[y0:y1, x0:x1]
    gl = fr.glow[y0:y1, x0:x1]
    a = (inside * fill_alpha)[..., None]
    img[:] = img * (1 - a) + a * np.asarray(fill, np.float32)
    flash = (inside * FL)[..., None] * C["cyan"]
    img[:] = img + flash * 0.55 + ghost[..., None] * C["cyan"] * 0.16
    gl[:] += flash * 0.6 + ghost[..., None] * C["cyan"] * 0.1
    landed = (K >= 0.999).astype(np.float32)
    e = (edge * landed)[..., None]
    img[:] = img * (1 - e) + e * np.asarray(line, np.float32)
    return float(smooth(seg(t, end - 0.1, end)))
