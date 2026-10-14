"""Design system v3 'cyber deep sea': palette, grounds, selective glow, glass panels, mixed-script type,
the TRACE panel (her thinking as code), hologram printing of a figure."""
import math
import re
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lib import H, W, blur, hexc, value_noise

C = dict(
    bg0=hexc("#03050b"), bg1=hexc("#08132a"),
    grid=hexc("#0f1d3a"), grid2=hexc("#16305e"),
    ink=hexc("#070c1d"), blue=hexc("#2459ff"), blue_dim=hexc("#14307f"), cyan=hexc("#3ee6ff"), ice=hexc("#e6f7ff"),
    amber=hexc("#ffb23e"), amber_dim=hexc("#7a4a14"), amber_ink=hexc("#2a1a08"),
    sys=hexc("#8592b3"), sys_dim=hexc("#3c4766"), text=hexc("#d7e3f7"),
    glass=hexc("#0a1122"), glass_line=hexc("#24447a"),
    violet=hexc("#8f6bff"),
)
FONT = dict(mono="C:/Windows/Fonts/CascadiaMono.ttf", sans="C:/Windows/Fonts/NotoSansSC-VF.ttf",
            serif="C:/Windows/Fonts/NotoSerifSC-VF.ttf")
_C = {}


class Frame:
    """An image plus an emission buffer: whatever is drawn 'neon' also lands in .glow and blooms at the end."""

    def __init__(self, img):
        self.img = img
        self.glow = np.zeros_like(img)

    def fill(self, m, color, amt=1.0, glow=0.0):
        c = np.asarray(color, np.float32)
        a = (m * amt)[..., None]
        self.img = self.img * (1 - a) + a * c
        if glow:
            self.glow += a * c * glow
        return self

    def add(self, m, color, amt=1.0, glow=0.0):
        c = np.asarray(color, np.float32)
        self.img = self.img + (m * amt)[..., None] * c
        if glow:
            self.glow += (m * amt)[..., None] * c * glow
        return self


def finish(fr, idx=0, glow_k=1.0, vignette=0.55, scan=0.05, ca=1.4, grain=0.018):
    g = fr.glow
    img = fr.img + (blur(g, 6) * 0.9 + blur(g, 22) * 0.7 + blur(g, 60) * 0.35) * glow_k
    if scan:
        if "scan" not in _C:
            s = np.ones((H, 1, 1), np.float32)
            s[::3] = 1 - scan
            _C["scan"] = s
        img = img * _C["scan"]
    if "vig" not in _C:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        _C["vig"] = (d ** 2.4)[..., None]
        r = np.random.default_rng(5)
        _C["grain"] = [r.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
    img = img * (1 - vignette * _C["vig"])
    if ca:
        for ch, s in ((0, 1 + ca * 0.0006), (2, 1 - ca * 0.0006)):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
            img[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    gn = cv2.resize(_C["grain"][idx % 6], (W, H))[..., None]
    img = img + gn * grain
    return np.clip(img, 0, 1)


# ------------------------------------------------------------------ grounds
def ground(key="base", glows=(), grid=True, seed=0):
    """Deep near-black blue, a fine grid with crosses on the majors, soft cool glows. Cached by key."""
    if key in _C:
        return _C[key].copy()
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = (C["bg1"] * (1 - y) + C["bg0"] * y) * np.ones((1, W, 1), np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    for (cx, cy, rx, ry, col, k) in glows:
        d = np.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
        img += (np.exp(-d * d * 2.2) * k)[..., None] * np.asarray(col, np.float32)
    if grid:
        g = np.zeros((H, W), np.float32)
        g[:, ::48] = 0.5
        g[::48, :] = 0.5
        g[:, ::192] = 1
        g[::192, :] = 1
        img = img * (1 - g[..., None] * 0.22) + g[..., None] * C["grid"] * 0.22
        cr = np.zeros((H, W), np.float32)
        for yy in range(0, H, 192):
            for xx in range(0, W, 192):
                cv2.line(cr, (xx - 6, yy), (xx + 6, yy), 1.0, 1)
                cv2.line(cr, (xx, yy - 6), (xx, yy + 6), 1.0, 1)
        img = img * (1 - cr[..., None] * 0.6) + cr[..., None] * C["grid2"] * 0.6
    n = value_noise(H, W, 300, seed=seed, octaves=3) - 0.5
    img += n[..., None] * 0.03
    _C[key] = img.astype(np.float32)
    return _C[key].copy()


def brackets(fr, x0, y0, x1, y1, color, L=18, amt=0.8, glow=0.0, w=1):
    m = np.zeros((H, W), np.float32)
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(m, (int(x), int(y)), (int(x + sx * L), int(y)), 1.0, w, cv2.LINE_AA)
        cv2.line(m, (int(x), int(y)), (int(x), int(y + sy * L)), 1.0, w, cv2.LINE_AA)
    return fr.fill(m, color, amt, glow)


def rrect(m, x0, y0, x1, y1, r, val=1.0):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2)))
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), val, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), val, -1)
    if r:
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(m, (cx, cy), r, val, -1, cv2.LINE_AA)
    return m


def glass(fr, x0, y0, x1, y1, r=14, alpha=0.78, line=None, hi=True, line_amt=0.9):
    m = rrect(np.zeros((H, W), np.float32), x0, y0, x1, y1, r)
    fr.fill(m, C["glass"], alpha)
    inner = rrect(np.zeros((H, W), np.float32), x0 + 1, y0 + 1, x1 - 1, y1 - 1, max(0, r - 1))
    fr.fill(np.clip(m - inner, 0, 1), line if line is not None else C["glass_line"], line_amt)
    if hi:
        top = np.zeros((H, W), np.float32)
        cv2.line(top, (int(x0 + r), int(y0) + 1), (int(x1 - r), int(y0) + 1), 1.0, 1)
        fr.fill(top, C["cyan"], 0.25)
    return fr


# ------------------------------------------------------------------ type (mixed scripts)
def _font(role, size):
    key = ("font", role, size)
    if key not in _C:
        f = ImageFont.truetype(FONT[role], size)
        if role in ("sans", "serif"):
            try:
                f.set_variation_by_axes([500 if role == "sans" else 600])
            except Exception:
                pass
        _C[key] = f
    return _C[key]


def _is_cjk(ch):
    return ord(ch) >= 0x2E80 or ch in "，。？！：；“”（）、…—"


@lru_cache(maxsize=4096)
def mtext(s, size, latin="mono", cjk="sans"):
    """Mask for a string mixing Latin (mono) and CJK (sans) on one baseline. Returns (mask, baseline_y)."""
    runs = []
    for ch in s:
        role = cjk if _is_cjk(ch) else latin
        if runs and runs[-1][0] == role:
            runs[-1][1] += ch
        else:
            runs.append([role, ch])
    asc = max(_font(r, size).getmetrics()[0] for r, _ in runs) if runs else size
    dsc = max(_font(r, size).getmetrics()[1] for r, _ in runs) if runs else 0
    wsum = int(sum(_font(r, size).getlength(t) for r, t in runs)) + 4
    im = Image.new("L", (max(1, wsum), asc + dsc + 4), 0)
    d = ImageDraw.Draw(im)
    x = 2
    for r, t in runs:
        f = _font(r, size)
        d.text((x, asc + 2), t, fill=255, font=f, anchor="ls")
        x += f.getlength(t)
    return np.asarray(im, np.float32) / 255, asc + 2


def put(fr, mask, x, y, color, amt=1.0, anchor="lt", glow=0.0):
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
        return fr
    m = (mask[ya - y0:yb - y0, xa - x0:xb - x0] * amt)[..., None]
    c = np.asarray(color, np.float32)
    fr.img[ya:yb, xa:xb] = fr.img[ya:yb, xa:xb] * (1 - m) + m * c
    if glow:
        fr.glow[ya:yb, xa:xb] += m * c * glow
    return fr


def label(fr, s, x, y, color, size=18, amt=1.0, anchor="lt", latin="mono", cjk="sans", glow=0.0):
    m, _ = mtext(s, size, latin, cjk)
    return put(fr, m, x, y, color, amt, anchor, glow)


# ------------------------------------------------------------------ the TRACE panel: her thinking, as code
KW = {"for", "in", "if", "return", "assert", "while", "not", "and", "or", "null", "true", "false", "self", "else"}
TOK = re.compile(r'(//.*$)|("[^"]*")|(\b\d+(?:\.\d+)?(?:Hz|k|%)?\b)|([A-Za-z_][A-Za-z_0-9\.]*)(?=\()|([A-Za-z_][A-Za-z_0-9\.]*)|(\s+)|(.)')


def tokens(line):
    """Colour roles for a line of pseudo-code: >, <, ! prefixes; keywords; calls; strings (the user's words); comments."""
    out = []
    if line[:2] in ("> ", "< ", "! "):
        out.append((line[:2], {">": "cyan", "<": "ice", "!": "amber"}[line[0]]))
        line = line[2:]
    for m in TOK.finditer(line):
        com, st, num, call, ident, ws, other = m.groups()
        if com:
            out.append((com, "com"))
        elif st:
            out.append((st, "amber"))
        elif num:
            out.append((num, "num"))
        elif call:
            out.append((call, "ice"))
        elif ident:
            out.append((ident, "cyan" if ident in KW else "text"))
        elif ws:
            out.append((ws, "text"))
        else:
            out.append((other, "sys"))
    return out


ROLE = dict(cyan=C["cyan"], ice=C["ice"], amber=C["amber"], com=C["sys"], num=hexc("#9fb4ff"), text=C["text"], sys=C["sys"])


@lru_cache(maxsize=2048)
def code_line_mask(line, size):
    """Render one coloured code line -> list of (mask, color_role, x_offset), total width, height."""
    parts = []
    x = 0
    hmax = 0
    for txt, role in tokens(line):
        if not txt:
            continue
        m, _ = mtext(txt, size, "mono", "sans")
        parts.append((m, role, x))
        x += m.shape[1] - 4
        hmax = max(hmax, m.shape[0])
    return parts, x, hmax


@lru_cache(maxsize=4096)
def line_strip(text_, ts, size, lh, w, dim, glow_last):
    """One rendered TRACE line: (rgb, alpha, glow) strips of height lh+8 and width w."""
    hh = lh + 8
    rgb = np.zeros((hh, w, 3), np.float32)
    a = np.zeros((hh, w), np.float32)
    g = np.zeros((hh, w, 3), np.float32)

    def blit(m, x, y, col, k=1.0, gk=0.0):
        sub = np.zeros((hh, w), np.float32)
        _paste(sub, m, x, y)
        sub *= k
        c = np.asarray(col, np.float32)
        rgb[:] = rgb * (1 - sub[..., None]) + sub[..., None] * c
        a[:] = np.maximum(a, sub)
        if gk:
            g[:] += sub[..., None] * c * gk
    m, _ = mtext(ts, size - 4, "mono", "sans")
    blit(m, 16, 5, C["sys_dim"])
    parts, wtot, _ = code_line_mask(text_, size)
    for pm, role, xo in parts:
        blit(pm, 86 + xo, 2, ROLE[role], dim, 0.35 if (glow_last and role in ("amber", "cyan")) else 0.0)
    return rgb, a, g, wtot


def trace_panel(fr, t, script, box, title="深度思考", size=19, lh=31, cps=46.0, alpha=1.0, frame=True):
    """script: list of (t_appear, line). Lines type in at cps chars/s with a decoding head; older lines dim; auto-scroll.
    frame=False when the panel's own box is being built by build.box_build()."""
    x0, y0, x1, y1 = [int(v) for v in box]
    if alpha <= 0:
        return fr
    if frame:
        glass(fr, x0, y0, x1, y1, r=12, alpha=0.93 * alpha)
    img = fr.img[y0:y1, x0:x1]
    gl = fr.glow[y0:y1, x0:x1]
    h, w = img.shape[:2]

    def lfill(mask, col, amt=1.0, g=0.0):
        a = (mask * amt * alpha)[..., None]
        c = np.asarray(col, np.float32)
        img[:] = img * (1 - a) + a * c
        if g:
            gl[:] += a * c * g

    hd = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(hd, np.array([[26, 23], [33, 30], [26, 37], [19, 30]], np.int32), 1.0, cv2.LINE_AA)
    lfill(hd, C["cyan"], 1.0, 0.6)
    sub = np.zeros((h, w), np.float32)
    _paste(sub, mtext(title, 20, "mono", "sans")[0], 44, 16)
    lfill(sub, C["text"])
    sub = np.zeros((h, w), np.float32)
    tm = mtext("TRACE", 15, "mono", "sans")[0]
    _paste(sub, tm, w - 24 - tm.shape[1], 19)
    lfill(sub, C["sys"])
    sep = np.zeros((h, w), np.float32)
    cv2.line(sep, (16, 56), (w - 16, 56), 1.0, 1)
    lfill(sep, C["glass_line"])
    shown = [(ta, ln) for ta, ln in script if t >= ta]
    if not shown:
        return fr
    avail = (h - 80) // lh
    n = len(shown)
    over = max(0, n - avail)
    scroll = (over - 1 + min(1.0, (t - shown[-1][0]) / 0.18)) if over > 0 else 0
    ty0 = 76 - scroll * lh
    top, bot = 60, h - 6
    for i, (ta, ln) in enumerate(shown):
        yy = int(round(ty0 + i * lh)) - 2
        if yy + lh + 8 < top or yy > bot:
            continue
        last = i == n - 1
        nch = len(ln) if not last else min(len(ln), int((t - ta) * cps) + 1)
        shown_txt = ln[:nch]
        if last and nch < len(ln):
            # the head of the line is still decoding
            from build import wrong_char
            r_ = np.random.default_rng(int(t * 30) + i)
            for ch in ln[nch:nch + 2]:
                shown_txt += wrong_char(ch, r_, "░▒▓█#%&$*+=<>/")
        rgb, a, g, wtot = line_strip(shown_txt, f"{ta:05.2f}", size, lh, w, 1.0 if i >= n - 3 else 0.62, last)
        ya, yb = max(top, yy), min(bot, yy + a.shape[0])
        if ya >= yb:
            continue
        sa = (a[ya - yy:yb - yy] * alpha)[..., None]
        img[ya:yb] = img[ya:yb] * (1 - sa) + rgb[ya - yy:yb - yy] * sa
        gl[ya:yb] += g[ya - yy:yb - yy] * alpha
        if last and (t % 0.5) < 0.3:
            car = np.zeros((h, w), np.float32)
            cx_ = 86 + wtot + 4
            cv2.rectangle(car, (int(cx_), int(yy + 4)), (int(cx_ + 9), int(yy + size + 4)), 1.0, -1)
            car[:top] = 0
            car[bot:] = 0
            lfill(car, C["cyan"], 1.0, 0.5)
    return fr


def _paste(dst, m, x, y):
    h, w = m.shape
    x0, y0 = int(round(x)), int(round(y))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if xa < xb and ya < yb:
        dst[ya:yb, xa:xb] = np.maximum(dst[ya:yb, xa:xb], m[ya - y0:yb - y0, xa - x0:xb - x0])


# ------------------------------------------------------------------ figures
def print_figure(fr, P, inks=None, rim=True, rim_col=None, shadow_off=(10, 7), alpha=1.0):
    """Solid printing (the sea): ink / blue / ice plates + an out-of-register deep plate + a thin neon rim."""
    dark, mid, light, fg = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    inks = inks or (C["ink"], C["blue"], C["ice"])
    if shadow_off:
        sh = cv2.warpAffine(fg, np.float32([[1, 0, shadow_off[0]], [0, 1, shadow_off[1]]]), (W, H))
        fr.fill(sh * alpha, C["blue_dim"], 0.9)
    for m, c in ((mid, inks[1]), (light, inks[2]), (dark, inks[0])):
        fr.fill(m * alpha, c)
    if rim:
        e = np.clip(fg - cv2.warpAffine(fg, np.float32([[1, 0, 0], [0, 1, 4]]), (W, H), borderMode=cv2.BORDER_REPLICATE), 0, 1)
        e = cv2.GaussianBlur(e, (0, 0), 0.8)
        fr.fill(e * alpha, rim_col if rim_col is not None else C["cyan"], 0.9, glow=0.8)
    return fr


def hologram(fr, P, t, color=None, alpha=1.0, jitter=0.0, seed=0):
    """The screen-side printing: the same plates as scanline light, with a glowing edge (a projection, not a body)."""
    color = C["cyan"] if color is None else color
    dark, mid, light, fg = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    inten = light * 1.0 + mid * 0.5 + dark * 0.12
    if jitter:
        r = np.random.default_rng(int(t * 24) + seed)
        rows = np.zeros(H, np.float32)
        for _ in range(4):
            y = int(r.uniform(0, H))
            rows[y:y + int(r.uniform(2, 14))] = r.uniform(-1, 1) * jitter
        mx = (np.arange(W, dtype=np.float32)[None, :] - rows[:, None]).astype(np.float32)
        my = np.broadcast_to(np.arange(H, dtype=np.float32)[:, None], (H, W)).astype(np.float32)
        inten = cv2.remap(inten, mx, my, cv2.INTER_LINEAR)
        fg = cv2.remap(fg, mx, my, cv2.INTER_LINEAR)
    lines = np.full((H, 1), 0.38, np.float32)
    lines[::3] = 1.0
    fr.fill(fg * alpha, C["ink"], 0.55)
    fr.fill(inten * lines * alpha, color, 0.95, glow=0.35)
    e = np.clip(fg - cv2.erode(fg, np.ones((3, 3), np.uint8)), 0, 1)
    fr.fill(e * alpha, C["ice"], 0.9, glow=0.9)
    return fr
