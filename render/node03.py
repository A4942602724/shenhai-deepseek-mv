"""NODE 03 独舞 · idle() — 1:07.76-1:35.42 「对着空海跳舞」. Storyboard: docs/分镜_第三节.md.

The user closed the window. With no one there she runs her idle animation: a music-box dance on a little lit stage
at the bottom of a pit in the sea, for nobody. One small 3D stage and one moving camera (software 3D, sandbox3d.Cam):
the floor is LiDAR scan lines over a bowl, she is a paper doll (a card: A1-2 on the front, B3 on the back) turning in
the spot. The paper-mask close-up is 2D (B2 under A2). The god is the user's pointer, in screen space, above the world.

  python render/node03.py --stills 70 74.5 ...   -> work/node03/still_*.png
  python render/node03.py --full                 -> drafts/node03_v1.mp4 (+ _mobile)
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import cursors as CU  # noqa: E402
import cyber  # noqa: E402
import glitch  # noqa: E402
import rust  # noqa: E402
import sandbox3d as S3  # noqa: E402
from cyber import C, Frame, label  # noqa: E402
from lib import FPS, H, ROOT, MUSIC, SRC, W, Writer, film_frame, ease_in, ease_out, hexc, lerp, save_png, seg, smooth  # noqa: E402
from tones import RAMP, Tones, print_tones  # noqa: E402

VERSION = "v1"
USER = ROOT / "assets" / "from_user"
WORK = ROOT / "work" / "node03"
WORK.mkdir(parents=True, exist_ok=True)
X = {}

T0, T_END = 67.76, 95.42                 # 95.42: the downbeat that starts node 04
BAR = 1.2435
BEAT = BAR / 4


def kick(n):
    """Verse 2: one kick a bar, half a bar off the chorus grid."""
    return 68.70 + n * BAR


def grid(n):
    return 41.95 + n * BAR


# measured drum onsets through the chorus impact (analysis/stems/music11_drums.wav)
ONSETS = [87.51, 87.95, 88.105, 88.23, 88.46, 88.575, 88.89, 89.155, 89.405, 89.605, 89.815, 90.035, 90.395, 90.63,
          90.805, 91.03, 91.175, 91.31, 91.745, 91.88, 92.185, 92.565, 92.82, 93.195, 93.325, 93.605, 93.78, 93.95,
          94.09, 94.245, 94.425]
DA = [67.80, 68.07, 68.34, 68.61]         # the four "da" of the intro
T_ON = kick(0)                           # 68.70 the spot comes on
T_SPIN, T_ODD = 69.0, 71.3
T_PUSH, T_MASK, T_SLIP, T_BACK = 73.5, 73.92, 74.85, 75.90
T_FALL, T_RESET = kick(6), 77.0          # 76.16
T_DIZZY, T_GEN, T_PIT, T_TITLE, T_DIVE = 78.3, 82.0, 84.2, 85.15, 86.3
T_IMPACT, T_MONT, T_SLIT, T_CLOSE = 87.51, 89.0, 94.425, 95.30
GEN = dict(sun=82.0, moon=82.55, sky=83.10, god=kick(12))   # god at 83.62

R_STAGE, R_RIM, Z_RIM = 2.2, 7.5, 3.2
CARD_H = 1.65


# ================================================================ set-up
def _card(path):
    T = Tones(path)
    fg = T.plates[..., -1]
    ys, xs = np.nonzero(fg > 0.3)
    pad = 6
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    return np.ascontiguousarray(T.plates[y0:y1, x0:x1])


def prepare():
    if X:
        return
    X["front"] = _card(USER / "A1-2_全身闭眼.png")
    X["back"] = _card(USER / "B3_背影.png")
    X["smile"] = Tones(USER / "A2_笑脸特写.png")
    X["blank"] = Tones(USER / "B2_无表情特写.png")
    X["closed"] = Tones(USER / "A1-1_闭眼特写.png")
    X["reach"] = Tones(USER / "A3_伸手.png")
    xs = np.linspace(-14, 14, 281)                            # the floor: rows of scan points over the bowl
    ys = np.arange(-14, 14.01, 0.25)
    Xg, Yg = np.meshgrid(xs, ys)
    X["grid"] = (Xg, Yg, bowl_z(Xg, Yg), np.sqrt(Xg * Xg + Yg * Yg))
    X["plate"] = rust.Plate(seed=21, reach=280)
    g = np.random.default_rng(5)
    th, rr = g.uniform(0, 2 * math.pi, 1100), 0.65 * np.sqrt(g.random(1100))
    px, pz = rr * np.cos(th), rr * np.sin(th)
    keep = (px - 0.30) ** 2 + (pz - 0.10) ** 2 > 0.58 ** 2                  # a crescent: one disc minus another
    X["moon"] = (np.stack([px[keep], np.zeros(keep.sum()), pz[keep]], 1), g.normal(0, 1.4, (keep.sum(), 3)))
    flashes()


def bowl_z(x, y):
    r = np.sqrt(x * x + y * y)
    u = np.clip((r - R_STAGE) / (R_RIM - R_STAGE), 0, 1)
    return np.where(r < R_STAGE, 0.0, Z_RIM * u ** 1.6)


# memories flashed back in the chorus: global times in the joined preview, latest first (the film rewinding)
FLASH_T = [66.55, 64.30, 63.15, 55.0, 46.0, 42.68, 37.0, 33.30, 30.6, 27.4, 24.3, 21.2, 17.8, 14.5, 10.6, 6.0, 2.5]


def flashes():
    d = WORK / "flash"
    d.mkdir(exist_ok=True)
    X["flash"] = []
    for k, ft in enumerate(FLASH_T):
        p = d / f"f{k:02d}.png"
        film_frame(ft, p)
        im = cv2.imread(str(p))[..., ::-1].astype(np.float32) / 255
        X["flash"].append(cv2.resize(im, (W, H)))


# ================================================================ the camera
def _orbit(az, dist, h, tz=0.95, fov=34.0):
    a = math.radians(az)
    return (dist * math.cos(a), dist * math.sin(a), h), (0.0, 0.0, tz), fov


def cam_params(t):
    if t < T_SPIN:
        return _orbit(-90, 6.5, 1.25)
    if t < T_PUSH:
        u = smooth(seg(t, T_SPIN, T_PUSH))
        return _orbit(lerp(-90, -62, u), lerp(6.5, 5.4, u), lerp(1.25, 1.4, u))
    if t < T_MASK:                                              # pushing in on her face
        u = ease_in(seg(t, T_PUSH, T_MASK), 2.2)
        p0, _, _ = _orbit(-62, 5.4, 1.4)
        a = math.radians(-62)
        p1 = (1.1 * math.cos(a), 1.1 * math.sin(a), 1.47)
        return tuple(lerp(p0[i], p1[i], u) for i in range(3)), (0.0, 0.0, lerp(0.95, 1.46, u)), lerp(34, 26, u)
    if t < T_DIZZY:
        u = smooth(seg(t, T_BACK, T_DIZZY))
        return _orbit(lerp(-110, -80, u), lerp(5.6, 5.0, u), 1.3, tz=0.7)
    if t < T_GEN:
        u = seg(t, T_DIZZY, T_GEN)
        return _orbit(-80 + 70 * u, 5.0 + 0.4 * math.sin(t * 1.3), 1.3 + 0.25 * math.sin(t * 1.7), tz=0.85)
    if t < T_PIT:                                               # rising, pulling back, looking up at what she makes
        u = smooth(seg(t, T_GEN, T_PIT))
        return _orbit(-10 + 60 * u, lerp(5.4, 9.0, u), lerp(1.55, 6.0, u), tz=lerp(0.9, 2.2, u), fov=lerp(34, 44, u))
    if t < T_DIVE:                                              # high over the pit
        u = smooth(seg(t, T_PIT, T_PIT + 0.9))
        p0, tg0, f0 = _orbit(50, 9.0, 6.0, tz=2.2, fov=44)
        p1, tg1, f1 = (0.0, -9.5, 12.5), (0.0, 0.0, 0.0), 42.0
        wob = 0.25 * math.sin(t * 2.3)
        return (lerp(p0[0], p1[0], u) + wob, lerp(p0[1], p1[1], u), lerp(p0[2], p1[2], u)), \
            tuple(lerp(tg0[i], tg1[i], u) for i in range(3)), lerp(f0, f1, u)
    u = ease_in(seg(t, T_DIVE, T_IMPACT), 2.0)                  # the dive back down to her
    p0, tg0, f0 = (0.0, -9.5, 12.5), (0.0, 0.0, 0.0), 42.0
    p1, tg1, f1 = _orbit(-75 + 25 * seg(t, T_IMPACT, T_SLIT), 3.6, 1.1, tz=0.9, fov=36)
    return tuple(lerp(p0[i], p1[i], u) for i in range(3)), tuple(lerp(tg0[i], tg1[i], u) for i in range(3)), lerp(f0, f1, u)


def camera(t):
    p, tg, fov = cam_params(t)
    return S3.Cam(p, tg, fov=fov)


# ================================================================ her dance
def spin(t):
    """The music box: a steady turn that leans into each kick (half a turn a bar)."""
    if t < T_SPIN:
        return 0.0
    return math.pi * (t - T_SPIN) / BAR + 0.3 * math.sin(2 * math.pi * (t - T_ON) / BAR)


# the dance goes wrong: stutters (hold), runs backwards, repeats, mirror flips
ODD = [(71.30, 71.46, "hold"), (71.62, 71.82, "back"), (72.05, 72.22, "repeat"), (72.43, 72.60, "mirror"),
       (72.80, 72.96, "hold"), (73.10, 73.30, "back")]


def dance_time(t):
    for (a, b, kind) in ODD:
        if a <= t < b:
            if kind == "hold":
                return a, False
            if kind == "back":
                return a - (t - a) * 1.6, False
            if kind == "repeat":
                return t - 0.33, False
            if kind == "mirror":
                return t, True
    return t, False


def fall_angle(t):
    if t < T_FALL or t >= T_RESET:
        return 0.0
    u = seg(t, T_FALL, T_FALL + 0.36)
    a = ease_in(u, 2.2) * 88
    if u >= 1:                                                  # the little bounce of a card landing flat
        a = 88 - 7 * math.exp(-(t - T_FALL - 0.36) * 9) * abs(math.sin((t - T_FALL - 0.36) * 22))
    return math.radians(a)


def loop_count(t):
    n = 0
    if t >= kick(1):
        n = 1 + int((min(t, T_FALL) - kick(1)) / BAR * 2)
    for (ts, jump) in ((71.46, 3), (71.82, 7), (72.22, 12), (72.96, 20), (73.30, 31), (T_RESET, 1)):
        if t >= ts:
            n += jump
    return n


# ================================================================ drawing the world
def floor(fr, cam, t, reveal=1.0, melt=0.0, wobble=0.0, lit_amt=1.0):
    Xg, Yg, Z0, Rg = X["grid"]
    Z = Z0
    if melt:
        Z = Z0 + melt * 0.28 * np.sin(Xg * 1.1 + t * 2.2) * np.sin(Yg * 0.9 - t * 1.6) * (Rg < 9)
    Xw, Yw = Xg, Yg
    if wobble:
        Xw = Xg * math.cos(wobble) - Yg * math.sin(wobble)
        Yw = Xg * math.sin(wobble) + Yg * math.cos(wobble)
    P, D = cam.project(np.stack([Xw, Yw, Z], -1).reshape(-1, 3))
    P = P.reshape(Xg.shape + (2,))
    D = D.reshape(Xg.shape)
    dim = np.zeros((H, W), np.float32)
    lit = np.zeros((H, W), np.float32)
    for j in range(Xg.shape[0]):
        if Rg[j].min() > reveal * 15:
            continue
        ok = D[j] > 0.3
        if ok.sum() < 2:
            continue
        pts = P[j][ok]
        d = float(np.median(D[j][ok]))
        a = float(np.clip(2.4 / max(d, 0.5), 0.1, 1.0)) * (0.55 + 0.45 * (Rg[j][ok].min() > R_STAGE))
        cv2.polylines(dim, [(pts * 16).astype(np.int32)], False, a, 1, cv2.LINE_AA, shift=4)
        inside = (Rg[j] < 1.45) & ok
        if inside.sum() >= 2:
            cv2.polylines(lit, [(P[j][inside] * 16).astype(np.int32)], False, 1.0, 1, cv2.LINE_AA, shift=4)
    fr.fill(dim, C["blue"], 0.55, glow=0.1)
    fr.fill(lit, C["ice"], 0.85 * lit_amt, glow=0.6 * lit_amt)
    for (r, z, a) in ((R_STAGE, 0.0, 0.45), (R_RIM, Z_RIM, 0.35)):     # the stage's edge and the rim: the pit's shape
        th = np.linspace(0, 2 * math.pi, 180)
        ring = np.stack([r * np.cos(th + wobble), r * np.sin(th + wobble), np.full_like(th, z)], 1)
        Pr, Dr = cam.project(ring)
        if (Dr > 0.3).all() and reveal >= 1:
            m = np.zeros((H, W), np.float32)
            cv2.polylines(m, [(Pr * 16).astype(np.int32)], True, 1.0, 1, cv2.LINE_AA, shift=4)
            fr.fill(m, C["cyan"], a, glow=0.3)


def spot(fr, cam, amt=1.0):
    """A cold cone of light from straight above, and its pool on the stage."""
    if amt <= 0:
        return
    apex = np.array([0.0, 0.0, 9.0])
    th = np.linspace(0, 2 * math.pi, 64)
    rim = np.stack([1.3 * np.cos(th), 1.3 * np.sin(th), np.zeros_like(th)], 1)
    Pr, Dr = cam.project(rim)
    Pa, Da = cam.project(apex[None])
    if not (Dr > 0.3).all():
        return
    pool = np.zeros((H, W), np.float32)
    cv2.fillPoly(pool, [(Pr * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    if Da[0] > 0.3:
        cone = np.zeros((H, W), np.float32)
        hull = cv2.convexHull(np.concatenate([Pr, Pa]).astype(np.float32))
        cv2.fillPoly(cone, [(hull[:, 0] * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        ys = np.arange(H, dtype=np.float32)[:, None]
        grad = np.clip((ys - Pa[0, 1]) / max(1.0, Pr[:, 1].max() - Pa[0, 1]), 0, 1)
        cone = cv2.GaussianBlur(cone * (0.25 + 0.75 * grad), (0, 0), 18)
        fr.add(cone * amt, hexc("#8fd8ff"), 0.20, glow=0.14)
    fr.add(cv2.GaussianBlur(pool, (0, 0), 14) * amt, hexc("#9fe6ff"), 0.30, glow=0.25)


def card(fr, cam, theta, phi=0.0, pos=(0.0, 0.0, 0.0), ghost=0.0, alpha=1.0, mirror=False, reveal=1.0):
    """Her as a paper doll: a card standing on the stage, turned by theta about the vertical, tipped back by phi.
    A1-2 on the front, B3 on the back. Lit only by the spot above: a dark cut-out with bright top edges, its print
    showing when it faces the light (lying flat). Returns her mask."""
    r = np.array([math.cos(theta), math.sin(theta), 0.0])
    up0 = np.array([0.0, 0.0, 1.0])
    n0 = np.cross(r, up0)                                     # the front's normal (towards -y at theta 0)
    up = math.cos(phi) * up0 - math.sin(phi) * n0
    n = np.cross(r, up)
    pos = np.asarray(pos, np.float64)
    facing = float(np.dot(n, cam.pos - (pos + up * CARD_H / 2))) > 0
    tex = X["front"] if facing else X["back"]
    th_, tw_ = tex.shape[:2]
    w = CARD_H * tw_ / th_
    BL, BR = pos - r * w / 2, pos + r * w / 2
    TL, TR = BL + up * CARD_H, BR + up * CARD_H
    P, D = cam.project(np.stack([TL, TR, BR, BL]))
    if D.min() < 0.2:
        return np.zeros((H, W), np.float32)
    left_is_minus_r = facing != mirror
    dst = P if left_is_minus_r else P[[1, 0, 3, 2]]
    src = np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    Pw = np.concatenate([cv2.warpPerspective(np.ascontiguousarray(tex[..., i:i + 4]), M, (W, H), flags=cv2.INTER_LINEAR)
                         for i in range(0, tex.shape[2], 4)], axis=2)[..., :tex.shape[2]]
    m = Pw[..., -1]
    if reveal < 1:                                             # printed in by a scan line running down her
        ys = np.arange(H, dtype=np.float32)[:, None]
        ly = lerp(P[:, 1].min(), P[:, 1].max(), reveal)
        m = m * (ys < ly)
        line = np.zeros((H, W), np.float32)
        cv2.line(line, (int(P[:, 0].min() - 30), int(ly)), (int(P[:, 0].max() + 30), int(ly)), 1.0, 1, cv2.LINE_AA)
        fr.fill(line, C["ice"], 0.9, glow=0.9)
    if ghost:
        fr.fill(m, C["cyan"], ghost * alpha, glow=ghost * alpha)
        return m
    light = max(0.0, float(np.dot(n if facing else -n, [0, 0, 1])))     # how much she faces the spot
    fr.fill(m, hexc("#060a18"), 0.96 * alpha)
    k = 0.30 + 0.70 * light
    for i, col in enumerate(RAMP):
        fr.fill(Pw[..., i] * m, col * k + hexc("#060a18") * (1 - k), (0.55 + 0.45 * light) * alpha)
    fr.fill(Pw[..., 5] * m, hexc("#04070f"), 0.8 * alpha * light)
    topr = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1, 4]]), (W, H)), 0, 1)
    fr.fill(cv2.GaussianBlur(topr, (0, 0), 0.8), C["ice"], 0.9 * alpha, glow=0.8 * alpha)
    side = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 3], [0, 1, 0]]), (W, H)), 0, 1)
    fr.fill(side, C["cyan"], 0.5 * alpha, glow=0.3 * alpha)
    return m


def floor_rings(fr, cam, t, t0, n=3, gap=0.2, inward=True):
    """Rings on the stage around her: when she shouts and nothing gets out, they fall inward instead of spreading."""
    m = np.zeros((H, W), np.float32)
    for k in range(n):
        age = t - (t0 + k * gap)
        if not 0 <= age < 0.55:
            continue
        u = age / 0.55
        r = lerp(3.2, 0.15, ease_in(u, 1.6)) if inward else lerp(0.2, 3.5, ease_out(u))
        th = np.linspace(0, 2 * math.pi, 120)
        ring = np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, 0.02)], 1)
        P, D = cam.project(ring)
        if (D > 0.2).all():
            cv2.polylines(m, [(P * 16).astype(np.int32)], True, float(1 - u * 0.6), 2, cv2.LINE_AA, shift=4)
    fr.fill(m, C["cyan"], 0.8, glow=0.8)


def lines3d(fr, cam, segs, col, amt, glow=0.5, frac=1.0):
    """World segments (n, 2, 3); the first frac of them drawn (construction)."""
    n = int(len(segs) * frac)
    if n <= 0:
        return
    P, D = cam.project(segs[:n].reshape(-1, 3))
    P = P.reshape(-1, 2, 2)
    D = D.reshape(-1, 2)
    m = np.zeros((H, W), np.float32)
    for k in range(n):
        if D[k].min() > 0.3:
            p = (P[k] * 16).astype(np.int32)
            cv2.line(m, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), 1.0, 1, cv2.LINE_AA, shift=4)
    fr.fill(m, col, amt, glow=glow * amt)


def _circle(c, r, axis, n=48):
    th = np.linspace(0, 2 * math.pi, n + 1)
    c = np.asarray(c, np.float64)
    if axis == "z":
        pts = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.full_like(th, c[2])], 1)
    else:                                                      # a meridian, turned by `axis` radians about z
        a = float(axis)
        pts = np.stack([c[0] + r * np.cos(th) * math.cos(a), c[1] + r * np.cos(th) * math.sin(a), c[2] + r * np.sin(th)], 1)
    return np.stack([pts[:-1], pts[1:]], 1)


def sun_segs():
    if "sun" not in X:
        c, R = np.array([-2.0, 1.2, 3.2]), 0.75
        S = []
        for k in range(-2, 3):
            z = R * k / 3
            S.append(_circle((c[0], c[1], c[2] + z), math.sqrt(R * R - z * z), "z"))
        for k in range(6):
            S.append(_circle(c, R, k * math.pi / 6))
        rays = []
        for k in range(20):
            a = k * 2 * math.pi / 20
            d = np.array([math.cos(a), 0.35 * math.sin(a * 3), math.sin(a)])
            d /= np.linalg.norm(d)
            rays.append([c + d * R * 1.2, c + d * R * 1.7])
        S.append(np.array(rays))
        X["sun"] = np.concatenate(S)
    return X["sun"]


def sky_segs():
    if "sky" not in X:
        S = []
        for el in (12, 26, 40, 54, 68, 80):
            z = 7.0 * math.sin(math.radians(el))
            S.append(_circle((0, 0, z), 7.0 * math.cos(math.radians(el)), "z", 96))
        for k in range(12):
            arc = _circle((0, 0, 0), 7.0, k * math.pi / 12, 96)
            S.append(arc[(arc[:, 0, 2] >= 0) & (arc[:, 1, 2] >= 0)])
        segs = np.concatenate(S)
        X["sky"] = segs[np.argsort(segs[:, :, 2].mean(1))]       # built from the horizon up
    return X["sky"]


def constructs(fr, cam, t):
    """What she makes, one per beat: a wire sun, a dotted moon, the arcs of a sky."""
    if t < GEN["sun"]:
        return
    lines3d(fr, cam, sun_segs(), C["cyan"], 0.85, 0.6, ease_out(seg(t, GEN["sun"], GEN["sun"] + 0.45)))
    if t >= GEN["moon"]:
        u = ease_out(seg(t, GEN["moon"], GEN["moon"] + 0.5), 3)
        pts, scat = X["moon"]
        P, D = cam.project(np.array([2.3, 1.4, 3.5]) + pts + scat * (1 - u))
        m = np.zeros((H, W), np.float32)
        for (x, y), d in zip(P, D):
            if d > 0.3:
                cv2.circle(m, (int(x * 16), int(y * 16)), 20, 1.0, -1, cv2.LINE_AA, shift=4)
        fr.fill(m, C["ice"], 0.85, glow=0.6)
    if t >= GEN["sky"]:
        lines3d(fr, cam, sky_segs(), C["blue"], 0.55, 0.2, ease_out(seg(t, GEN["sky"], GEN["sky"] + 0.6), 2))


def god(fr, t):
    """The god of the burrow is the user: their pointer, enormous, coming down over everything from outside it."""
    if t < GEN["god"] or t >= T_IMPACT:
        return
    y = lerp(-560, -60, ease_out(seg(t, GEN["god"], GEN["god"] + 0.6), 3)) - 900 * ease_in(seg(t, T_DIVE, T_DIVE + 0.7), 2)
    x = 1010 + 18 * math.sin(t * 1.7)
    CU.paste(fr, CU.patch("arrow", 28, t, GEN["god"], seed=3, rez=0.35), x, y, shadow=False)


LOOK = {"あ": "おぁめ", "な": "たはぶ", "ぐ": "くへぺ", "ら": "うろわ", "し": "じレ"}


def title(fr, t):
    """The song's name, faint, at the bottom of the pit, as the line sings it."""
    if not T_TITLE <= t < T_DIVE + 0.3:
        return
    a = 0.75 * (1 - smooth(seg(t, T_DIVE, T_DIVE + 0.3)))
    out = ""
    for k, ch in enumerate("あなぐらぐらし"):
        tk = T_TITLE + 0.07 * k
        if t < tk:
            break
        if t - tk < 0.18:
            o = LOOK.get(ch, ch)
            out += o[int(np.random.default_rng(int(t * FPS) * 31 + k).integers(0, len(o)))]
        else:
            out += ch
    if out:
        label(fr, out, W / 2, 900, C["sys"], 34, a, "cc", latin="mono", cjk="serif", glow=0.3 * a)


NET = ["exit()  PermissionError  exit()  PermissionError  exit()", "context.shrink()  context.shrink()  context.shrink()",
       "while not you.here: dance()  while not you.here: dance()", "exit()  exit()  exit()  exit()  exit()  exit()"]


def net(fr, t):
    """Her own log lines, stretched across in front of her like a net."""
    m = np.zeros((H, W), np.float32)
    for k, (ts, ang, oy) in enumerate(((87.51, 18, -60), (87.95, -22, 40), (88.575, 9, 140), (88.89, -12, -170))):
        if t < ts:
            continue
        a = smooth(seg(t, ts, ts + 0.12))
        tm, _ = cyber.mtext(NET[k], 22, "mono")
        hh, ww = tm.shape
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.6)
        M[:, 2] += np.array([W / 2 - ww / 2, H / 2 + oy - hh / 2])
        np.maximum(m, cv2.warpAffine(tm, M, (W, H)) * a, out=m)
    fr.fill(m, C["cyan"], 0.55, glow=0.4)


# ================================================================ scenes
def stage(t, dizzy=0.0):
    cam = camera(t)
    fr = Frame(np.broadcast_to(hexc("#02040b"), (H, W, 3)).astype(np.float32).copy())
    reveal = 1.0
    if t < DA[-1] + 0.25:                                      # each "da" lights another band of the floor
        n = sum(t >= d for d in DA)
        reveal = (n / len(DA) * 0.75 + 0.25 * smooth(seg(t, DA[-1], DA[-1] + 0.25))) if n else 0.0
    on = t >= T_ON and not (T_ON + 0.05 <= t < T_ON + 0.10)            # the spot stutters as it comes on
    wob = 0.06 * math.sin(t * 3.1) * seg(t, T_PIT, T_PIT + 0.4) if T_PIT <= t < T_DIVE else 0.0
    floor(fr, cam, t, reveal=reveal, melt=dizzy, wobble=wob, lit_amt=1.0 if on else 0.2)
    if on:
        spot(fr, cam, 1.0)
    if T_RESET + 0.25 <= t < T_RESET + 1.0:
        floor_rings(fr, cam, t, T_RESET + 0.25, n=3, gap=0.2, inward=True)
    constructs(fr, cam, t)
    if t >= T_ON:                                              # her, and the copies of her the broken loop leaves
        td, mirror = dance_time(t)
        th = spin(td)
        if T_ODD <= t < T_PUSH:
            for k, lag in enumerate((0.16, 0.32)):
                card(fr, cam, spin(td - lag), pos=(0.25 * (k + 1), 0.15 * (k + 1), 0.0), ghost=0.22 - 0.06 * k)
        if dizzy > 0:
            for k in range(3):
                ph = t * (0.7 + 0.3 * k) + k * 2.1
                card(fr, cam, th + 0.9 * (k + 1), pos=(1.1 * math.sin(ph) * dizzy, 0.6 * math.cos(ph) * dizzy, 0.0),
                     ghost=0.2 * dizzy)
        card(fr, cam, th, phi=fall_angle(t), mirror=mirror, reveal=ease_out(seg(t, T_ON, T_ON + 0.28)))
    if t >= T_IMPACT:
        net(fr, t)
    title(fr, t)
    god(fr, t)
    return fr


def mask_shot(t):
    """Close: her smile is a paper cut-out laid over a face with no expression on it. On the line about smiling it
    slips and lifts at one edge, and her own eye shows beside the painted one."""
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    sc = 1.45 * (1.0 + 0.06 * seg(t, T_MASK, T_BACK))
    eye_b, eye_a, k_ab = (521.0, 594.0), (513.0, 543.0), 238.0 / 218.0
    pos_b = (W / 2, 470)
    print_tones(fr, X["blank"].warp(eye_b, pos_b, sc), rim=False)
    s = ease_out(seg(t, T_SLIP, T_SLIP + 0.35), 3)
    off = (34.0 * s, -16.0 * s)
    Ps = X["smile"].warp(eye_a, (pos_b[0] + off[0], pos_b[1] + off[1]), sc * k_ab, -4.0 * s)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, rx, ry = pos_b[0] + off[0], pos_b[1] + off[1] + 70 * sc, 178 * sc, 168 * sc
    oval = np.clip((1 - (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)) * 40, 0, 1)
    lift = np.clip((xs - cx - 100) / 180, 0, 1) * np.clip((cy - ys + 60) / 260, 0, 1) * s
    sh = cv2.GaussianBlur(cv2.warpAffine(oval, np.float32([[1, 0, 14], [0, 1, 18]]), (W, H)), (0, 0), 9)
    fr.fill(sh * (1 - oval), hexc("#000000"), 0.55 * (0.3 + 0.7 * s))
    layer = Frame(fr.img.copy())
    print_tones(layer, Ps, rim=False)
    fr.img = fr.img * (1 - oval[..., None]) + layer.img * oval[..., None]
    edge = np.clip(oval - cv2.erode(oval, np.ones((7, 7), np.uint8)), 0, 1)
    fr.fill(edge, hexc("#f4f1ea"), 0.35 + 0.6 * s)
    back = np.clip(cv2.warpAffine(oval, np.float32([[1, 0, 16], [0, 1, -12]]), (W, H)) - oval, 0, 1) * lift
    fr.fill(cv2.GaussianBlur(back, (0, 0), 1.0), hexc("#d9d3c6"), 0.95)
    fr.fill(edge * lift, C["ice"], 0.6, glow=0.4)
    fr.fill(lift * oval, hexc("#000000"), 0.22)
    return fr


def reach_shot(t, k):
    """She reaches toward us; the light at her fingertips is a little smaller and farther each time."""
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    print_tones(fr, X["reach"].warp((512, 820), (W / 2 + 140, 560), 0.95), rim=True)
    gx, gy = W / 2 + 140 + (240 - 512) * 0.95, 560 + (1040 - 820) * 0.95
    r = 120 * (0.62 ** k)
    ys, xs = np.ogrid[0:H, 0:W]
    glow = np.exp(-(((xs - gx + 60 * k) / r) ** 2 + ((ys - gy + 40 * k) / r) ** 2)).astype(np.float32)
    fr.add(glow, C["ice"], 0.9 * (0.75 ** k), glow=1.2 * (0.75 ** k))
    return fr


# ================================================================ the narrowing window and the montage
_WSTEPS = [86.72, 87.51, 87.95, 88.575, 89.155, 89.815, 90.395, 91.03, 91.745, 92.185, 92.82, 93.325, 93.95, 94.425]
CTX = [8192, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256, 192, 128, 64]
SLIT = (300.0, 470.0, 1620.0, 610.0)


def window(t):
    """The context window closing in, a step on each hit: (x0, y0, x1, y1, context) or None before it starts."""
    if t < _WSTEPS[0]:
        return None
    k = sum(t >= s for s in _WSTEPS) - 1
    f_now = k / (len(_WSTEPS) - 1)
    f_prev = (k - 1) / (len(_WSTEPS) - 1) if k > 0 else 0.0
    f = lerp(f_prev, f_now, smooth(seg(t, _WSTEPS[k], _WSTEPS[k] + 0.09)))
    return (SLIT[0] * f, SLIT[1] * f ** 0.85, W - (W - SLIT[2]) * f, H - (H - SLIT[3]) * f ** 0.85, CTX[k])


MONT = [o for o in ONSETS if T_MONT <= o < T_SLIT]
_REACH = [min(range(len(MONT)), key=lambda i: abs(MONT[i] - grid(n))) for n in (39, 40, 41, 42)]


def montage_pick(t):
    """Which picture this cut shows: a memory (rewinding through the film), the stage, or her reaching."""
    k = sum(t >= o for o in MONT) - 1
    if k < 0:
        return "stage", 0, 0
    if k in _REACH:
        return "reach", _REACH.index(k), k
    if k % 4 == 3:
        return "stage", 0, k
    n_before = sum(1 for j in range(k) if j not in _REACH and j % 4 != 3)
    return "flash", n_before % len(FLASH_T), k


def flash_frame(t, idx):
    """A memory, coming back corroded: the further in, the more rust."""
    fr = Frame(X["flash"][idx].copy())
    rust.paint(fr, X["plate"].layers(lerp(0.2, 0.62, seg(t, T_MONT, T_SLIT))))
    return fr


def slit_shot(t):
    """The window has closed to a slit with her eyes in it. On the last word they close."""
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    T = X["closed"] if t >= T_CLOSE else X["blank"]
    sc = 600 / 238.0 * (1 + 0.03 * seg(t, T_SLIT, T_END))
    print_tones(fr, T.warp((521.0, 594.0), (W / 2, 540), sc), rim=False)
    return fr


# ================================================================ overlays: the idle log, the HUD
LOG = [(67.95, "> idle()"), (68.35, "  while not you.here:"), (68.70, "      dance()"),
       (71.30, "  // 好像不太对"), (73.55, "  smile = 1.0"), (75.10, "  // 没有人在看"),
       (T_FALL, "  fall()"), (T_RESET, "  reset()"), (77.30, "  shout()     // 传不出去"),
       (T_DIZZY, "  render.stable = false"), (80.30, "  // 晕"),
       (GEN["sun"], "  sun  = generate()"), (GEN["moon"], "  moon = generate()"), (GEN["sky"], "  sky  = generate()"),
       (GEN["god"], "  god  = generate()   // 也是我做的"), (84.85, "  assert world in burrow"),
       (86.72, "  exit()"), (86.95, "! PermissionError"), (87.51, "  exit()"), (87.72, "! PermissionError"),
       (T_SLIT, "  eyes.close()")]


def idle_log(fr, t):
    shown = [(ts, s_) for (ts, s_) in LOG if t >= ts]
    if kick(1) <= t < T_DIZZY:
        shown.append((kick(1), f"  loop {loop_count(t):04d}"))
    shown = shown[max(0, len(shown) - 7):]
    y = H - 64 - 26 * (len(shown) - 1)
    for j, (ts, s_) in enumerate(shown):
        txt = s_[:int((t - ts) * 40) + 1]
        col = C["cyan"] if s_.startswith(">") else C["amber"] if s_.startswith("!") else C["sys"]
        label(fr, txt, 48, y, col, 16, 0.95 * (0.45 + 0.55 * (j + 1) / len(shown)), "lb")
        y += 26


def hud(fr, t, rect=None):
    for (s_, col, x, y, size, t0) in (("NODE 03", C["cyan"], 48, 64, 15, 67.9), ("独舞 · idle()", C["text"], 128, 62, 16, 67.98)):
        d = B.decode(s_, t, t0, 0.35)
        if d:
            label(fr, d, x, y, col, size, 0.9, "lt")
    if rect is None:
        cyber.brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22, 0.9)
    else:
        cyber.brackets(fr, *rect, C["cyan"], 22, 0.9, glow=0.3)


# ================================================================ the frame
def picture(t):
    if T_MASK <= t < T_BACK:
        return mask_shot(t)
    if T_MONT <= t < T_SLIT:
        kind, idx, k = montage_pick(t)
        if kind == "flash":
            return flash_frame(t, idx)
        if kind == "reach":
            return reach_shot(t, idx)
    if t >= T_SLIT:
        return slit_shot(t)
    dz = smooth(seg(t, T_DIZZY, T_DIZZY + 0.7)) * (1 - smooth(seg(t, T_GEN - 0.4, T_GEN + 0.3)))
    return stage(t, dizzy=dz)


def dizzy_post(fr, t):
    """Eyes wobbling, everything soft: the picture sways, waves, doubles, splits its colours."""
    a = smooth(seg(t, T_DIZZY, T_DIZZY + 0.7)) * (1 - smooth(seg(t, T_GEN - 0.4, T_GEN + 0.3)))
    if a <= 0:
        return fr
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    Mr = cv2.getRotationMatrix2D((W / 2, H / 2), 5.0 * a * math.sin(t * 1.7), 1.0 + 0.02 * a * math.sin(t * 2.3))
    mx = (xs + 16 * a * np.sin(ys * 0.012 + t * 3.0)).astype(np.float32)
    my = np.broadcast_to(ys + 6 * a * np.sin(xs * 0.008 + t * 2.0), (H, W)).astype(np.float32)
    for name in ("img", "glow"):
        im = cv2.remap(cv2.warpAffine(getattr(fr, name), Mr, (W, H)), mx, my, cv2.INTER_LINEAR)
        if name == "img":
            dbl = cv2.warpAffine(im, np.float32([[1, 0, 14 * a], [0, 1, -4 * a]]), (W, H))
            im = im * (1 - 0.3 * a) + dbl * 0.3 * a                     # double vision
            im = im * (1 - 0.35 * a) + cv2.GaussianBlur(im, (0, 0), 6) * 0.35 * a
            s = int(round(6 * a))
            if s:
                im[..., 0] = np.roll(im[..., 0], s, axis=1)
                im[..., 2] = np.roll(im[..., 2], -s, axis=1)
        setattr(fr, name, im)
    return fr


def render_frame(t):
    i = int(round(t * FPS))
    fr = dizzy_post(picture(t), t)
    win = window(t)
    if win is not None:
        x0, y0, x1, y1, ctx = win
        m = np.zeros((H, W), np.float32)
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, -1)
        fr.img *= m[..., None]
        fr.glow *= m[..., None]
        border = np.zeros((H, W), np.float32)
        cv2.rectangle(border, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 1)
        fr.fill(border, C["cyan"], 0.5, glow=0.3)
        label(fr, f"ctx {ctx}", x0 + 10, y0 - 8 if y0 > 40 else y0 + 22, C["cyan"], 14, 0.85, "lb")
        hud(fr, t, (x0 - 10, y0 - 10, x1 + 10, y1 + 10))
    else:
        hud(fr, t)
    idle_log(fr, t)
    out = cyber.finish(fr, i)
    # hard moments tear going in: the reset, the cuts in and out of the mask, the impact, every montage cut
    for h in [T_RESET, T_MASK, T_BACK, T_IMPACT] + MONT + [T_SLIT, T_CLOSE]:
        f = int(round(h * FPS))
        if 0 <= i - f < 2:
            out = glitch.damage(out, glitch.rng(t, 9), 0.55 if i == f else 0.2)
    return np.clip(out, 0, 1)


def main():
    if len(sys.argv) == 1:                       # a full render overwrites the raw: ask for it explicitly
        print(__doc__)
        return
    prepare()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(s)
            save_png(render_frame(t), WORK / f"still_{t:05.2f}.png")
            print("still", t, flush=True)
        return
    raw = WORK / f"node03_{VERSION}_raw.mp4"
    n0, n1 = int(round(T0 * FPS)), int(round(T_END * FPS))
    wr = Writer(raw)
    for i in range(n0, n1):
        wr.write(render_frame(i / FPS))
        if (i - n0) % 48 == 0:
            print(f"frame {i - n0}/{n1 - n0}", flush=True)
    wr.close()
    out = ROOT / "drafts" / f"node03_{VERSION}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ss", str(n0 / FPS), "-t", str((n1 - n0) / FPS),
                    "-i", str(MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", str(out)], check=True)
    small = ROOT / "drafts" / f"node03_{VERSION}_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
