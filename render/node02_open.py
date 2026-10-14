"""NODE 02 opening, v2 (32.62-41.95). Storyboard: docs/分镜_第二节.md, N2-01 v2.

She closes her eyes and the frame closes like an eyelid. Black. Then, out of order and damaged, her singing comes
back in flashes: close-ups of her profile mid-note (P set), each 1-3 frames, torn, split, on plates that are
rusting a little more every time. Between them, and then for good, the present: the deep, where she sings and a
ping goes out on every tick of the music. Nothing comes back.

No TRACE panel in this stretch. The HUD returns, decoded, once the picture settles.
"""
import math

import cv2
import numpy as np

import build as B
import cyber
import glitch
import rust
from cyber import C, Frame, brackets, label
from lib import FPS, H, ROOT, W, Snow, ease_in, hexc, lerp, seg, smooth
from tones import Tones, print_tones

PDIR = ROOT / "assets" / "from_user" / "P_侧脸唱歌"
FILES = dict(P1="P1_侧脸唱啊.png", P2="P2_侧脸唱喔.png", P3="P3_侧脸唱咿.png", P4="P4_侧脸闭口哼.png",
             P5="P5_闭眼侧脸唱.png", P6="P6_仰头唱高音.png")
MOUTH = dict(P1=(988, 277), P2=(1003, 293), P3=(963, 246), P4=(994, 241), P5=(978, 573), P6=(1023, 215))

BAR = 1.2435
BEAT = BAR / 4
DROP = 41.95
T_EYES = 32.60                                  # her eyes close (node 01's last picture swaps to closed eyes)
T_LID0, T_LID1, T_BLACK = 32.62, 32.83, 32.90   # the frame closes like an eyelid, then black
T_FIRST = 33.24                                 # the first sound after the silence
T_SETTLE = 33.41                                # from here the present holds between flashes
T_HUD = 36.19
EYE_Y, EYE_X = 425, 1250
TICKS = [33.70, 34.32, 34.95, 35.57, 36.19, 36.81, 37.43]          # the offbeat "嘀" of the build
BEATS_UP = [DROP - 3 * BAR + k * BEAT for k in range(12)]          # 38.22 ... 41.64, every beat
PINGS = TICKS + BEATS_UP
SEA_MOUTH = (1040, 590)

O = {}


def prepare():
    if O:
        return
    O["T"] = {k: Tones(PDIR / f) for k, f in FILES.items()}
    O["plate"] = rust.Plate(seed=4)
    O["snow"] = Snow(300, seed=21, vy=(-70, -25))
    r = np.random.default_rng(33)
    n = 46
    O["flakes"] = dict(x=r.uniform(0, W, n), y=r.uniform(0, H, n), vy=r.uniform(-34, -12, n), vx=r.uniform(-6, 6, n),
                       s=r.uniform(2.0, 6.5, n) ** 1.1, ph=r.uniform(0, 6.3, n), w=r.uniform(1.2, 3.2, n),
                       shape=[r.uniform(0.55, 1.0, 5) for _ in range(n)], hot=r.random(n) < 0.3)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    O["xs"], O["ys"] = xs, ys


# ================================================================ the present: the deep, singing
def sea_cam(t):
    """Scale and mouth position of the wide shot: a slow push-in that quickens with the build."""
    k = smooth(seg(t, 34.0, 38.22)) * 0.05 + ease_in(seg(t, 38.22, 41.64), 2) * 0.17
    return 1.0 + k, (SEA_MOUTH[0], SEA_MOUTH[1] + 6 * math.sin(t * 0.9))


def sway(P, t, m):
    """Hair and the fin-ear drift in the water; the face stays still."""
    xs, ys = O["xs"], O["ys"]
    A = np.clip((m[0] - 260 - xs) / 420, 0, 1) * 15 + np.clip((ys - m[1] - 230) / 300, 0, 1) * np.clip((m[0] + 40 - xs) / 420, 0, 1) * 9
    dx = A * np.sin(ys * 0.010 + t * 1.2 + xs * 0.003)
    dy = A * 0.35 * np.sin(xs * 0.008 + t * 0.9)
    mx, my = xs + dx, ys + dy
    return np.concatenate([cv2.remap(np.ascontiguousarray(P[..., i:i + 4]), mx, my, cv2.INTER_LINEAR) for i in range(0, P.shape[2], 4)], axis=2)[..., :P.shape[2]]


def flakes(fr, t):
    """A few flakes of rust drifting up with the snow, tumbling, catching what light there is."""
    F = O["flakes"]
    m_lo = np.zeros((H, W), np.float32)
    m_hi = np.zeros((H, W), np.float32)
    for i in range(len(F["x"])):
        x = (F["x"][i] + F["vx"][i] * t + 9 * math.sin(t * 0.5 + F["ph"][i])) % W
        y = (F["y"][i] + F["vy"][i] * t) % H
        s = F["s"][i]
        tumble = abs(math.cos(t * F["w"][i] + F["ph"][i])) * 0.8 + 0.2
        ang = t * F["w"][i] * 0.7 + F["ph"][i]
        pts = []
        for k, rr in enumerate(F["shape"][i]):
            a = ang + k * 2 * math.pi / 5
            pts.append((x + math.cos(a) * s * rr, y + math.sin(a) * s * rr * tumble))
        poly = (np.array(pts) * 16).astype(np.int32)
        cv2.fillPoly(m_hi if F["hot"][i] else m_lo, [poly], 1.0, cv2.LINE_AA, shift=4)
    fr.fill(m_lo, rust.RUST["body"], 0.85)
    fr.fill(m_hi, rust.RUST["hot"], 0.9)
    return fr


def pings(fr, t, m, fg):
    """Every tick of the music a ping goes out from her mouth into the empty water. No echo returns."""
    ring = np.zeros((H, W), np.float32)
    for tp in PINGS:
        age = t - tp
        if age < 0 or age > 1.7:
            continue
        r = 70 + 1000 * age ** 0.9
        a = (1 - age / 1.7) ** 1.6
        for dr, k in ((0, 1.0), (-16, 0.4)):
            rr = r + dr
            if rr > 10:
                cv2.ellipse(ring, (int(m[0] * 16), int(m[1] * 16)), (int(rr * 16), int(rr * 16)), 0, -58, 58, a * k, 2,
                            cv2.LINE_AA, shift=4)
    ring *= 1 - fg
    fr.add(ring, C["cyan"], 0.85, glow=0.9)
    return fr


def sea(t, hud=True, notes=True):
    depth = seg(t, T_FIRST, DROP)
    g = cyber.ground("sea2", [(1000, -80, 1300, 560, C["blue_dim"], 0.8), (1600, 600, 800, 520, hexc("#0a1838"), 0.5)],
                     grid=False)
    fr = Frame(g * (1 - 0.4 * depth))
    fr.fill(O["snow"].mask(t), C["ice"], 0.18 + 0.14 * depth)
    flakes(fr, t)
    s, m = sea_cam(t)
    P = sway(O["T"]["P5"].warp(MOUTH["P5"], m, 1.05 * s), t, m)
    print_tones(fr, P)
    pings(fr, t, m, P[..., -1])
    # a small annotation riding beside the first rings
    for (txt, dy, t0) in ((("f₀  52 Hz", 0, 35.57), ("echo  —", 26, 37.43)) if notes else ()):
        d = B.decode(txt, t, t0, 0.3)
        if d:
            label(fr, d, m[0] + 190, m[1] - 250 + dy * 1.15, C["sys"], 17, 0.9, "lt")
    if hud and t >= T_HUD:
        hud_(fr, t)
    return fr


def hud_(fr, t):
    a = smooth(seg(t, T_HUD, T_HUD + 0.2))
    for (s_, col, x, y, size, t0) in (("NODE 02", C["cyan"], 48, 64, 15, T_HUD), ("对话 · respond()", C["text"], 128, 62, 16, T_HUD + 0.08)):
        d = B.decode(s_, t, t0, 0.35)
        if d:
            label(fr, d, x, y, col, size, 0.9, "lt")
    d = B.decode("INST", t, T_HUD + 0.15, 0.25)
    if d:
        label(fr, d, W - 48, H - 64, C["sys"], 15, 0.9, "rt")
    brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22 * a, 0.9)
    return fr


# ================================================================ the past: close-ups of her singing, rusting
VIEWS = dict(a1=("P1", 1.38, (1170, 500), 0), a1x=("P1", 1.85, (1060, 540), -4),
             e3=("P3", 1.55, (1130, 470), 2), o2=("P2", 1.95, (1060, 540), 0), h4=("P4", 1.45, (1140, 470), 0),
             hi6=("P6", 1.32, (1170, 420), 0), hi6x=("P6", 1.9, (1080, 500), 3), p5=("P5", 1.75, (1120, 560), 0))


def rust_at(t):
    """The memories corrode a little more every time they come back."""
    return lerp(0.16, 0.58, seg(t, T_FIRST, DROP))


def closeup(view, t, drift=0.0):
    key, s, pos, rot = VIEWS[view]
    s *= 1 + 0.035 * drift                                   # a held flashback creeps slowly closer
    fr = Frame(np.broadcast_to(C["bg0"] * 0.6, (H, W, 3)).astype(np.float32).copy())
    P = O["T"][key].warp(MOUTH[key], pos, s, rot)
    print_tones(fr, P)
    rust.paint(fr, O["plate"].layers(rust_at(t)))
    return fr


def black(t):
    return Frame(np.broadcast_to(C["bg0"] * 0.35, (H, W, 3)).astype(np.float32).copy())


# (start time, frames, what, damage). what: "black" | "sea" | a VIEWS key | ("mix", a, b).
# Flashbacks hold for a while (glitched going in and out, steadier in the middle).
EV = [
    # 32-37: two flashes only (review 2026-10-14): one as the node cuts in, one more on a tick; the deep holds between
    (33.24, 4, "a1", 0.9), (33.41, 3, "sea", 0.5),
    (35.57, 5, "hi6", 0.6),
    # 37-41: the build keeps its flashes; none from 41 s on
    (38.22, 5, "e3", 0.4), (38.84, 5, "o2", 0.45), (39.46, 5, "a1", 0.5), (40.08, 5, "h4", 0.5),
    (40.71, 4, "e3", 0.55),
]
_EVF = [(int(round(ts * FPS)), n, what, dmg) for (ts, n, what, dmg) in EV]


def envelope(k, n, dmg):
    """Damage over a held flash: a hard hit going in, steadier while it holds, a tear going out."""
    if n <= 2:
        return dmg
    if k == 0:
        return dmg
    if k == n - 1:
        return dmg * 0.75
    return dmg * 0.14


def flash(view, t, kn, dmg):
    """One finished frame of a held flashback: frame k of n."""
    k, n = kn
    img = cyber.finish(closeup(view, t, drift=k / FPS * 6), int(round(t * FPS)))
    return np.clip(glitch.damage(img, glitch.rng(t, 13), envelope(k, n, dmg)), 0, 1)


def event_at(i):
    """The event covering frame i, and the residual damage trailing the last flash."""
    last_end, last_dmg = -99, 0.0
    for (f0, n, what, dmg) in _EVF:
        if f0 <= i < f0 + n:
            return what, (envelope(i - f0, n, dmg) if isinstance(what, str) and what in VIEWS else dmg), i - f0
        if f0 + n <= i:
            last_end, last_dmg = f0 + n, dmg
    trail = i - last_end
    t = i / FPS
    default = "sea" if t >= T_SETTLE else "black"
    return default, (last_dmg * 0.45 if trail < 2 else 0.0), 0


def source(what, t, k=0):
    if what == "black":
        fr = black(t)
    elif what == "sea":
        fr = sea(t)
    else:
        fr = closeup(what, t, drift=k / FPS * 6)
    return cyber.finish(fr, int(round(t * FPS)))


def mix(a, b, r):
    out = a.copy()
    y = 0
    while y < H:
        h = int(r.choice([8, 16, 24, 40, 64, 96, 140]))
        if r.random() < 0.5:
            out[y:y + h] = b[y:y + h]
        y += h
    return out


def frame(t):
    """A finished frame for T_BLACK <= t < DROP."""
    i = int(round(t * FPS))
    what, dmg, k = event_at(i)
    r = glitch.rng(t, 11)
    if isinstance(what, tuple):
        img = mix(source(what[1], t), source(what[2], t), r)
    else:
        img = source(what, t, k)
    if what != "black" and dmg > 0:
        img = glitch.damage(img, r, dmg)
    return np.clip(img, 0, 1)


# ================================================================ the eyelid
def eyelid(img, t):
    """Node 01's held frame closes like an eye: lids from above and below meet on her eye line; a thin line of
    light lingers there a moment, then black."""
    if t < T_LID0:
        return img
    k = ease_in(seg(t, T_LID0, T_LID1), 2.0)
    top, bot = EYE_Y * k, H - (H - EYE_Y) * k
    ys = np.arange(H, dtype=np.float32)[:, None]
    fe = 46 * (1 - k) + 1.5
    m = np.clip((ys - top) / fe, 0, 1) * np.clip((bot - ys) / fe, 0, 1)
    out = img * m[..., None]
    if t >= T_LID1 - 0.05:
        la = 1 - smooth(seg(t, T_LID1, T_BLACK))
        xs = np.arange(W, dtype=np.float32)[None, :]
        line = np.exp(-((ys - EYE_Y) / 1.6) ** 2) * np.exp(-((xs - EYE_X) / 520) ** 2)
        out = out + (line * la)[..., None] * C["ice"] * 0.9
        out = out + (cv2.GaussianBlur(line, (0, 0), 6) * la * 2.2)[..., None] * C["cyan"]
    return np.clip(out, 0, 1)
