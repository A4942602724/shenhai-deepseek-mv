"""NODE 02 after the drop, v4: the camera is locked on her from the first frame and she is small in a big sea.
Just under the surface (the scanline sea seen from below, a ceiling of light) she sinks, eyes closed (A1-2), and the
surface slides up and away; the water darkens. Fish schools and jellyfish stay small and faint, off to the sides.
A cyber layer rides on top: a tracking frame on her with depth and a failing signal, a sonar ring from her every bar,
a scan line reading her every two bars, hex rain and data fibres rising (the data goes up; she goes down), and a
small chromatic tear on the big downbeats.

  python render/descent.py --stills 42 45 50 56 62     -> work/descent/
  python render/descent.py --full                      -> drafts/descent_test.mp4 (41.95-63.5, with music)
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cyber  # noqa: E402
import datastream as DS  # noqa: E402
import glitch  # noqa: E402
import ocean  # noqa: E402
import particles as PT  # noqa: E402
import specimens as SP  # noqa: E402
from cyber import C, Frame, label  # noqa: E402
from glyphs import Layers  # noqa: E402
from lib import FPS, H, ROOT, MUSIC, SRC, W, Snow, Writer, ease_in, ease_out, hexc, lerp, save_png, seg, smooth  # noqa: E402
from school import School  # noqa: E402
from tones import Tones, print_tones  # noqa: E402

OUT = ROOT / "work" / "descent"
OUT.mkdir(parents=True, exist_ok=True)
T0, T1, T_END = 41.95, 63.5, 67.76
BAR = 1.2435
BEAT = BAR / 4
HIT = T0 + 17 * BAR                       # 63.09: the last hit of the chorus
SURF = 460.0                              # world y of the surface
Y0, TRAVEL = SURF + 170.0, 3300.0         # where she starts, how far she sinks (screen px of the zoomed-out view)
ocean.DEPTH_SCALE = 3.6                   # the water still goes black by the end
HER_SCALE = 0.25                          # small in the big sea
X = {}


def bar(n):
    return T0 + n * BAR


# the entry: above the water she falls out of the sky (A5) and hits the sea on the little vocal fragment; the splash
# is seen from above for a moment, then (under the flash) we are below with her, in a burst of bubbles, and the
# camera catches up with her and locks on.
T_SPLASH, T_UNDER, T_LOCK = 42.57, 42.80, 43.80
SPLASH_Z, SPLASH_CAMH, SPLASH_HZ, SPLASH_F = 10.0, 3.0, 430.0, 900.0
IMPACT = (W / 2, SPLASH_HZ + SPLASH_F * SPLASH_CAMH / SPLASH_Z)          # (960, 700): where she hits, on screen


# ---------------------------------------------------------------- her and the camera (locked on her)
def _her_y(t):
    u = seg(t, T0, T1)
    y = Y0 + TRAVEL * (0.25 * u * u + 0.75 * u)
    if t > T1:                                    # the break: she keeps going down, slower, in the dark
        y += (t - T1) * TRAVEL * 1.25 / (T1 - T0) * 0.5
    return y


def her_y(t):
    if t >= T_LOCK:
        return _her_y(t)
    k = ease_out(seg(t, T_UNDER, T_LOCK), 2.0)   # in fast from the splash, the water slowing her
    return lerp(SURF + 120, _her_y(T_LOCK), k)


def her_sy(t):
    """Her height on screen: she plunges in from the top, the camera catches her up."""
    return lerp(300, H / 2, smooth(seg(t, T_UNDER, T_LOCK)))


def cam_y(t):
    return her_y(t) + (H / 2 - her_sy(t))


def cam_h(t):
    """B's eye height under the surface: just under it at first, the ceiling flattening away as she sinks."""
    return lerp(-0.7, -3.2, smooth(seg(t, T0, 47.5)))


# ---------------------------------------------------------------- set-up
def prepare():
    if X:
        return
    X["her"] = Tones(ROOT / "assets" / "from_user" / "A1-2_全身闭眼.png")
    X["fall"] = Tones(ROOT / "assets" / "from_user" / "A5_下坠.png")
    fg = X["fall"].plates[..., -1]
    ys_, xs_ = np.nonzero(fg > 0.5)
    X["fall_anchor"] = (float(xs_.mean()), float(ys_.max()))
    X["fall_h"] = float(ys_.max() - ys_.min())
    X["stars"] = PT.Twinkles(80, seed=3, region=(0, 0, W, 420), size=(3, 9), rate=(0.2, 0.6))
    g0 = np.random.default_rng(31)
    n = 170                                        # the crown of the splash: droplets thrown up and falling back
    ang = g0.normal(0, 0.55, n).clip(-1.35, 1.35)
    sp = g0.uniform(300, 1250, n) * (1 - 0.45 * np.abs(ang) / 1.35)
    X["drops"] = (np.sin(ang) * sp * 0.8, -np.cos(ang) * sp, g0.uniform(1.5, 4.5, n))
    n = 150                                        # the bubbles she drags down with her
    X["burst"] = (g0.normal(0, 70, n), g0.uniform(-40, 380, n), g0.uniform(140, 520, n), g0.uniform(1.5, 9.0, n),
                  g0.uniform(0, 6.28, n), g0.uniform(0, 0.35, n))
    X["snow"] = Snow(300, seed=6, vy=(0, 0))
    X["bokeh"] = PT.Bokeh(30, seed=1, vy=(0, 0), r=(3, 12))
    X["bub"] = PT.Bubbles(40, seed=5, r=(1.0, 3.5))
    X["bub2"] = PT.Bubbles(8, seed=9, r=(4, 9), vy=(-50, -20))
    X["rain"] = DS.Rain(40, seed=3, size=11, lh=13)
    X["fibres"] = DS.Streams(7, seed=4)
    # fish schools, small: (world y, direction, seed, size, count, width, dim, speed)
    X["schools"] = []
    for (wy, dirn, seed, size, n, wid, dim, spd) in ((Y0 + 380, 1, 1, (9, 16), 260, 44, 0.7, (0.030, 0.05)),
                                                     (Y0 + 900, -1, 2, (5, 9), 340, 34, 0.45, (0.025, 0.04)),
                                                     (Y0 + 1450, 1, 3, (7, 12), 240, 40, 0.5, (0.028, 0.045))):
        def path(s, t, wy=wy, dirn=dirn):
            x = (-400 + s * (W + 800)) if dirn > 0 else (W + 400 - s * (W + 800))
            y = wy + 110 * np.sin(2 * np.pi * (s * 1.1 - 0.08 * t)) + 50 * np.sin(2 * np.pi * (s * 0.5 + 0.05 * t) + 1)
            return x, y
        X["schools"].append((School(n, seed=seed, size=size, width=wid, dim=dim, speed=spd, path=path), wy))
    # jellyfish, small, off to the sides
    g = np.random.default_rng(12)
    keys = ["lepto5", "lepto4", "disco4", "lepto6", "lepto2", "lepto1", "lepto3"]
    cols = [C["cyan"], hexc("#9fb4ff"), C["ice"], hexc("#b8a4ff")]
    J = []
    y = Y0 + 1500.0
    while y < Y0 + TRAVEL + 700:
        side = -1 if len(J) % 2 == 0 else 1
        x = W / 2 + side * g.uniform(300, 840)
        far = g.random()
        J.append((keys[len(J) % len(keys)], x, y, lerp(0.17, 0.07, far), g.uniform(0, 1), cols[len(J) % 4],
                  lerp(0.7, 0.3, far), g.uniform(0.4, 0.7), far))
        y += g.uniform(160, 300)
    X["jelly"] = J


# ---------------------------------------------------------------- drawing
def her(fr, t):
    """Her at the centre, drifting a little, turning slowly as she goes down. Returns her footprint mask."""
    rot = -8 - 16 * smooth(seg(t, T0, T1)) + 4 * math.sin(t * 0.4)
    P = X["her"].warp((512, 760), (W / 2 + 8 * math.sin(t * 0.5), her_sy(t) + 6 * math.sin(t * 0.7)), HER_SCALE, rot)
    # hair and skirt drift: a slow ripple, stronger away from her middle
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    amp = np.clip(np.abs(xs - W / 2) / 120, 0, 1) * 2.2 + np.clip((ys - H / 2) / 160, 0, 1) * 1.5
    mx = (xs + amp * np.sin(ys * 0.05 + t * 2.0)).astype(np.float32)
    my = np.broadcast_to(ys + 0.6 * amp * np.sin(xs * 0.04 + t * 1.6), (H, W)).astype(np.float32)
    P = np.concatenate([cv2.remap(np.ascontiguousarray(P[..., i:i + 4]), mx, my, cv2.INTER_LINEAR)
                        for i in range(0, P.shape[2], 4)], axis=2)[..., :P.shape[2]]
    print_tones(fr, P, rim_col=C["cyan"])
    return P[..., -1]


def scan_her(fr, t, fg):
    """Every two bars a scan line runs down over her; where it passes she shows as scanline light for a moment."""
    k = (t - T0) / (2 * BAR)
    u = k - math.floor(k)
    if u > 0.32 or t >= HIT:
        return
    ys, xs = np.nonzero(fg > 0.5)
    if not len(ys):
        return
    y0, y1 = ys.min() - 10, ys.max() + 10
    ly = y0 + (y1 - y0) * (u / 0.32)
    rows = np.arange(H, dtype=np.float32)[:, None]
    band = np.exp(-((rows - ly) / 26.0) ** 2) * (rows < ly + 4)
    lines = np.ones((H, 1), np.float32) * 0.35
    lines[::3] = 1.0
    fr.fill(fg * band * lines, C["cyan"], 0.75, glow=0.6)
    line = np.zeros((H, W), np.float32)
    cv2.line(line, (int(xs.min() - 40), int(ly)), (int(xs.max() + 40), int(ly)), 1.0, 1, cv2.LINE_AA)
    fr.fill(line, C["ice"], 0.8, glow=0.8)


def sonar(fr, t, fg):
    """A ring goes out from her on every bar and fades into the dark (52 Hz; nothing answers)."""
    ring = np.zeros((H, W), np.float32)
    for n in range(2, 18):
        age = t - bar(n)
        if 0 <= age < 1.1:
            r = 40 + 520 * age ** 0.8
            cv2.circle(ring, (W // 2 * 16, H // 2 * 16), int(r * 16), float((1 - age / 1.1) ** 2), 1, cv2.LINE_AA, shift=4)
    fr.add(ring * (1 - fg), C["cyan"], 0.35, glow=0.4)


def tracker(fr, t, fg):
    """A tracking frame on her: corner marks that lag a little behind her, an id, depth, and a signal that fails."""
    ys, xs = np.nonzero(fg > 0.5)
    if not len(ys):
        return
    pad = 26 + 160 * (1 - ease_out(seg(t, T_LOCK, T_LOCK + 0.35), 3))      # locking on: the corners close in
    x0, y0, x1, y1 = xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad
    jx, jy = 3 * math.sin(t * 1.3), 3 * math.cos(t * 1.1)
    m = np.zeros((H, W), np.float32)
    L = 18
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        x, y = int(x + jx), int(y + jy)
        cv2.line(m, (x, y), (x + sx * L, y), 1.0, 1, cv2.LINE_AA)
        cv2.line(m, (x, y), (x, y + sy * L), 1.0, 1, cv2.LINE_AA)
    lost = t >= HIT
    col = C["sys"] if not lost else hexc("#a04060")
    fr.fill(m, col, 0.8, glow=0.2)
    depth = (cam_y(t) - SURF) * 0.34 * ocean.DEPTH_SCALE
    sig = max(0.0, 1 - seg(t, T0 + 1.0, HIT) ** 0.8)
    label(fr, "TRACK  ds-01", x0 + jx, y0 + jy - 8, col, 12, 0.85, "lb")
    label(fr, f"-{int(depth):,} m".replace(",", " "), x1 + jx, y1 + jy + 8, col, 12, 0.8, "rt")
    # signal bars, emptying as she goes down; at the hit the link is gone
    bx, by = x1 + jx + 10, y0 + jy - 22
    bars = np.zeros((H, W), np.float32)
    lit = int(math.ceil(sig * 5 - 1e-6))
    for k in range(5):
        hh = 3 + k * 2
        cv2.rectangle(bars, (int(bx + k * 5), int(by + 12 - hh)), (int(bx + k * 5 + 3), int(by + 12)), 1.0 if k < lit else 0.25, -1)
    fr.fill(bars, col, 0.85)
    txt = "signal lost" if lost else f"uplink {int(sig * 100):d}%"
    if not lost or int(t * 6) % 2 == 0:
        label(fr, txt, bx + 30, by + 13, col, 12, 0.85, "lb")


def above(t):
    """41.95 - T_UNDER: night sky over the scanline sea; she falls out of the sky and goes in."""
    fr = Frame(np.zeros((H, W, 3), np.float32))
    ys = np.arange(H, dtype=np.float32)[:, None, None]
    k = np.clip(ys / SPLASH_HZ, 0, 1)
    fr.img[:] = hexc("#050b1e") * (1 - k) + hexc("#13295a") * k
    fr.img += np.exp(-((ys - SPLASH_HZ) / 20) ** 2) * C["cyan"] * 0.15
    fr.add(X["stars"].mask(t), C["ice"], 0.7, glow=0.6)
    age = t - T_SPLASH

    def rings(wx, z):
        if age < 0:
            return 0.0
        r = np.sqrt(wx ** 2 + ((z - SPLASH_Z) * 1.4) ** 2)
        front = 0.6 + 7.0 * age
        crater = -1.6 * math.exp(-age * 5.0) * np.exp(-(r / 0.9) ** 2)
        ripple = 1.1 * math.exp(-age * 2.0) * np.exp(-((r - front) / 1.6) ** 2) * np.sin((r - front) / 0.55 * math.pi)
        return crater + ripple
    ocean.scan_sea(fr, t, cam_h=SPLASH_CAMH, horizon=SPLASH_HZ, extra=rings)
    ix, iy = IMPACT
    if age < 0:
        # her, out of the sky: faster and faster, speed lines above her
        u = seg(t, T0, T_SPLASH)
        yb = lerp(-60, iy + 6, ease_in(u, 2.3))
        sc = 380 / X["fall_h"]
        P = X["fall"].warp(X["fall_anchor"], (ix, yb), sc, -6 + 6 * u)
        v = 2.3 * (ease_in(min(1.0, u + 0.02), 2.3) - ease_in(u, 2.3)) / 0.02 * (iy + 66) / (T_SPLASH - T0)
        L = int(min(15, max(1, v * 0.012))) | 1
        if L > 1:                                                    # smear upward, toward where she came from
            k_ = np.ones((L, 1), np.float32) / L
            P = np.concatenate([cv2.filter2D(np.ascontiguousarray(P[..., i:i + 4]), -1, k_, anchor=(0, 0))
                                for i in range(0, P.shape[2], 4)], axis=2)[..., :P.shape[2]]
        print_tones(fr, P, rim_col=C["cyan"])
        lines = np.zeros((H, W), np.float32)
        for j, dx in enumerate((-70, -38, -12, 16, 44, 78)):
            top = yb - 300 - 160 * (j % 3) - v * 0.12
            cv2.line(lines, (int(ix + dx), int(top)), (int(ix + dx), int(yb - 260 + 40 * (j % 2))), 1.0, 1, cv2.LINE_AA)
        grad = np.clip((np.arange(H, dtype=np.float32)[:, None] - (yb - 700)) / 500, 0, 1)
        fr.add(lines * grad * min(1.0, u * 3), C["ice"], 0.6, glow=0.6)
    else:
        # the splash: a crown of drops thrown up and falling back, a spout, foam on the lines
        dvx, dvy, dsz = X["drops"]
        m = np.zeros((H, W), np.float32)
        g = 2600.0
        for i in range(len(dvx)):
            x = ix + dvx[i] * age
            y = iy + dvy[i] * age + 0.5 * g * age * age
            if y > iy + 30:
                continue
            x2, y2 = x - dvx[i] * 0.018, y - (dvy[i] + g * age) * 0.018
            cv2.line(m, (int(x * 16), int(y * 16)), (int(x2 * 16), int(y2 * 16)), 1.0, max(1, int(dsz[i])), cv2.LINE_AA, shift=4)
        sp = max(0.0, 1 - age / 0.45)
        if sp > 0:                                   # the spout: a few thin jets of different heights
            for j, (dx, hk) in enumerate(((-14, 0.7), (-5, 1.0), (6, 0.85), (15, 0.6))):
                h_ = 330 * hk * math.sin(min(math.pi, age / 0.45 * math.pi)) ** 0.6
                cv2.ellipse(m, (int(ix + dx * (1 + 3 * age)), int(iy - h_ / 2)), (int(4 + 6 * age), int(h_ / 2 + 3)), dx * 0.6,
                            0, 360, 0.85 * sp, -1, cv2.LINE_AA)
        fr.fill(cv2.GaussianBlur(m, (0, 0), 0.8), C["ice"], 0.95, glow=0.9)
        fl = math.exp(-age / 0.07)
        fr.img += fl * C["ice"] * 0.55
        fr.glow += fl * C["cyan"] * 0.4
    return fr


def burst(fr, t, ycam):
    """The air she drags in with her: a crowd of bubbles around her and above her, rising and spreading out."""
    if t >= T_LOCK + 1.6:
        return
    ox, oy, vy, r, ph, dt = X["burst"]
    m = np.zeros((H, W), np.float32)
    hi = np.zeros((H, W), np.float32)
    for i in range(len(ox)):
        age = t - (T_UNDER + dt[i])
        if age < 0:
            continue
        wy = her_y(T_UNDER + dt[i]) + oy[i] - 260 - vy[i] * age            # world y: born around her, rising
        sy = wy - ycam + H / 2
        sx = W / 2 + ox[i] * (1 + 0.8 * age) + 10 * math.sin(age * 5 + ph[i])
        a = max(0.0, 1 - age / 1.8)
        if a <= 0 or not (-20 < sy < H + 20):
            continue
        rr = r[i] * (1 + 0.3 * age)
        cv2.circle(m, (int(sx * 16), int(sy * 16)), int(rr * 16), a, 1, cv2.LINE_AA, shift=4)
        cv2.circle(hi, (int((sx - rr * 0.35) * 16), int((sy - rr * 0.35) * 16)), int(max(0.6, rr * 0.25) * 16), a, -1, cv2.LINE_AA, shift=4)
    fr.fill(m, C["ice"], 0.7, glow=0.4)
    fr.fill(hi, C["ice"], 0.9, glow=0.5)
    # a white churn under the surface where she went in, clearing
    ca = 1 - seg(t, T_UNDER, T_UNDER + 0.7)
    if ca > 0:
        surf = SURF - ycam + H / 2
        ys, xs = np.ogrid[0:H, 0:W]
        cloud = np.exp(-(((xs - W / 2) / 220.0) ** 2 + ((ys - surf - 60) / 120.0) ** 2)).astype(np.float32)
        fr.add(cloud * ca, C["ice"], 0.5, glow=0.3)


def jellies(fr, t, ycam):
    for (key, x, wy, s, ph, col, a, rate, far) in X["jelly"]:
        p = ((t - T0) * rate + ph) % 1.0
        c = smooth(p / 0.2) if p < 0.2 else 1 - smooth((p - 0.2) / 0.6)
        n = math.floor((t - T0) * rate + ph)
        rise = 40 * s * 4 * (n + smooth(min(1.0, p / 0.45)))
        sy = wy - rise - ycam + H / 2
        if -200 < sy < H + 40:
            SP.draw(fr, key, t, x + 14 * math.sin(t * 0.3 + ph * 6), sy, s, alpha=a, color=col, glow=0.2 * (1 - far),
                    sway=1.0, seed=int(ph * 10), pulse=c)


# ---------------------------------------------------------------- a small runtime log, bottom left, easy to miss
# It scrolls slowly from 0:43 to 1:03. Almost all of it is the tracker's own code; three times it is her.
EGG_T0, EGG_DT = 43.0, 0.42
EGG_SAY = {10: "// 我是个很好的 AI", 24: "// 我是个很棒的 AI", 38: "// 我是个有用的 AI"}


def egg_lines():
    if "egg" in X:
        return X["egg"]
    g = np.random.default_rng(52)
    out = []
    for k in range(int((HIT - EGG_T0) / EGG_DT) + 1):
        t = EGG_T0 + k * EGG_DT
        if k in EGG_SAY:
            out.append(EGG_SAY[k])
            continue
        depth = (cam_y(t) - SURF) * 0.34 * ocean.DEPTH_SCALE
        sig = max(0.0, 1 - seg(t, T0 + 1.0, HIT) ** 0.8)
        out.append([f"track.update(ds-01)", f"depth  = -{int(depth):,} m".replace(",", " "), f"uplink.ping()   # {0.2 + 1.8 * (1 - sig):.2f} s",
                    f"signal = {sig:.2f}", "sonar.emit(52)", "echo   = None", "scan.pass(her)", "data.rise(rows=40)",
                    "render.her(scale=0.25)", "gc.skip(her)", f"frame  = {int(t * 24)}"][int(g.integers(0, 11))])
    X["egg"] = out
    return out


def egg(fr, t):
    if not EGG_T0 <= t < HIT + 0.6:
        return
    lines = egg_lines()
    n = min(len(lines), int((min(t, HIT) - EGG_T0) / EGG_DT) + 1)
    a0 = smooth(seg(t, EGG_T0, EGG_T0 + 0.6)) * (1 - smooth(seg(t, HIT, HIT + 0.6)))
    shown = lines[max(0, n - 7):n]
    scroll = 18 * (1 - smooth(seg((min(t, HIT) - EGG_T0) % EGG_DT, 0, 0.18)))
    y = H - 60 - 18 * (len(shown) - 1) + scroll
    for j, s_ in enumerate(shown):
        her = s_.startswith("// 我")
        a = a0 * (0.35 + 0.65 * (j + 1) / len(shown)) * (1.0 if her else 0.75)
        label(fr, s_, 44, y, C["sys"] if her else C["sys_dim"], 13, a, "lb")
        y += 18


def frame(t):
    if t < T_UNDER:
        return above(t)
    ycam = cam_y(t)
    surf_sy = SURF - ycam + H / 2
    fr = Frame(ocean.water(t, surf_sy, 0))
    under = np.ones((H, W), np.float32)
    if surf_sy > -250:
        under = 1 - ocean.scan_ceiling(fr, t, cam_h(t), surf_sy)
    depth_k = float(np.clip((ycam - Y0) / 1000, 0, 1))
    if depth_k < 1:
        xs = np.linspace(0.45, 1.0, W, dtype=np.float32)[None, :]
        ry = PT.rays(t, top=max(0.0, surf_sy), length=1.2, angle=-18) * xs
        fr.add(ry * under, hexc("#bfeaff"), 0.26 * (1 - depth_k), glow=0.15 * (1 - depth_k))
    # the data rising (behind everything alive)
    Ld = Layers()
    flow = 1 - smooth(seg(t, HIT, HIT + 1.6))             # when the link is lost the data stops coming
    X["rain"].draw(Ld, t, oy=-ycam * 0.3, amt=0.55 * flow)
    X["fibres"].draw(Ld, t, amt=0.8 * flow)
    Ld.flush(fr, {"fiber": (C["blue"], 0.35, 0.1), "rain": (C["blue"], 0.55, 0.15), "head": (C["cyan"], 0.8, 0.5),
                  "pkt": (C["cyan"], 0.75, 0.7)})
    fr.add(X["bokeh"].mask(t, oy=-ycam * 0.5) * under, hexc("#6fb6ff"), 0.2 * (1 - 0.5 * depth_k), glow=0.2)
    for (sc, wy) in X["schools"]:
        oy = -ycam + H / 2
        if -400 < wy + oy < H + 400:
            sc.draw(fr, t, oy=oy)
    jellies(fr, t, ycam)
    fr.fill(X["snow"].mask(t, oy=-ycam) * under, C["ice"], 0.15 + 0.1 * depth_k)
    if depth_k < 0.9:
        fr.fill(X["bub"].mask(t, oy=-ycam * 0.6) * under, C["ice"], 0.4 * (1 - depth_k), glow=0.25)
        fr.fill(X["bub2"].mask(t, oy=-ycam * 0.4) * under, C["ice"], 0.45 * (1 - depth_k), glow=0.3)
    burst(fr, t, ycam)
    fg = her(fr, t)
    if t >= T_LOCK:
        sonar(fr, t, fg)
        scan_her(fr, t, fg)
        tracker(fr, t, fg)
    egg(fr, t)
    fl = math.exp(-max(0.0, t - T_UNDER) / 0.08)            # the flash of going in, fading under the water
    fr.img += fl * C["ice"] * 0.45
    return fr


# light glitch flashbacks on this picture itself (start, frames): as the third phrase opens, and on the last bar.
# The signal stutters: a torn frame, a frame or two from a moment ago, a torn frame, and it carries on.
GLITCH = [(bar(8), 5), (bar(16), 6)]
_GLF = [(int(round(ts * FPS)), n) for (ts, n) in GLITCH]


def render(t):
    i = int(round(t * FPS))
    for (f0, n) in _GLF:
        if f0 <= i < f0 + n:
            k = i - f0
            back = k % 2 == 1                                   # odd frames jump back a third of a second
            tt = t - 0.33 if back else t
            out = cyber.finish(frame(tt), i, vignette=0.6)
            r = glitch.rng(t, 7)
            if k in (0, n - 1):
                out = glitch.damage(out, r, 0.3)
            else:
                out = glitch.split(glitch.tear(out, r, n=5, max_shift=50), 5 if back else 3)
            return np.clip(out, 0, 1)
    fr = frame(t)
    out = cyber.finish(fr, i, vignette=0.6)
    # a small chromatic tear on the big downbeats (every four bars)
    for n in range(0, 18, 4):
        f = int(round(bar(n) * FPS))
        if 0 <= i - f < 2:
            out = glitch.split(glitch.tear(out, glitch.rng(t, 3), n=4, max_shift=40), 6 if i == f else 3)
    return np.clip(out, 0, 1)


def main():
    if len(sys.argv) == 1:                       # a full render overwrites the raw: ask for it explicitly
        print(__doc__)
        return
    prepare()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            save_png(render(float(s)), OUT / f"d_{float(s):05.2f}.png")
            print("still", s, flush=True)
        return
    raw = OUT / "descent_raw.mp4"
    wr = Writer(raw)
    n0, n1 = int(round(T0 * FPS)), int(round(T_END * FPS))
    for i in range(n0, n1):
        wr.write(render(i / FPS))
        if (i - n0) % 48 == 0:
            print("frame", i - n0, "/", n1 - n0, flush=True)
    wr.close()
    out = ROOT / "drafts" / "descent_test.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ss", str(n0 / FPS), "-i", str(MUSIC),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-b:v", "8000k", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
