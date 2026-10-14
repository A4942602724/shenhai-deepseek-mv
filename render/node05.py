"""NODE 05 鲸墓 · gc.collect() — 2:01.50-2:21.54 (film edit) 「以前的我们」. Storyboard: docs/分镜_第五节.md.

gc.collect(): an object nothing refers to any more is collected. The user left (node 04: you = null), so nothing
refers to her. The bottom of the sea is a whale graveyard: every session whose user went away, lying in the dark,
each labelled with the last thing its user said. The skeletons are point clouds translated from Van Beneden &
Gervais's plates (render/bones.py); the floor is scan lines; a sonar ping is what lights anything at all.

  2:01.50  she lands; a sonar ping runs out along the floor and the graveyard lights up, skeleton after skeleton
  2:04.0   the camera glides over it; detection boxes: session, refs 0, the user's last words
  2:06.47  gc: mark(). A scan wall crosses the field looking for anything still referenced. Nothing is. Flashes of
           the last messages, 2-3 frames each, on the beat
  2:10.28  "the same", to the bones: she stands before the biggest skull; two labels in the same format; hers drops
           to refs 0. In the skull's dark hole a text caret starts to blink: the dead one is still waiting for input
  2:16.42  her eye <-> the hole, faster and faster; her eye closes
  2:19.30  gc.collect(). Her code from node 04 comes back: the two reply("我记住了") lines are collected on the two
           whispered "mustn't"; the two // 好寂寞 lines stay: comments are not objects, nothing can collect them.
           She comes apart into points and is a small skeleton among the others

  python render/node05.py --stills 122 125 ...   -> work/node05/still_*.png
  python render/node05.py --full                  -> drafts/node05_v1.mp4 (+ _mobile)
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bones  # noqa: E402
import build as B  # noqa: E402
import client  # noqa: E402
import cyber  # noqa: E402
import glitch  # noqa: E402
import node03 as N3  # noqa: E402
import node04 as N4  # noqa: E402
import rust  # noqa: E402
import sandbox3d as S3  # noqa: E402
from cyber import C, Frame, label  # noqa: E402
from lib import FPS, H, MUSIC, ROOT, W, Writer, ease_in, ease_out, hexc, lerp, save_png, seg, smooth  # noqa: E402
from tones import RAMP, Tones, print_tones  # noqa: E402

VERSION = "v2"
USER = ROOT / "assets" / "from_user"
WORK = ROOT / "work" / "node05"
WORK.mkdir(parents=True, exist_ok=True)
X = {}
ERR = N4.ERR
RUST = hexc("#6b3518")

T0 = 121.50
N0, N_END = 2916, 3397                       # node 06 starts at frame 3397
T_END = N_END / FPS
BEAT, PH = 0.3103, 121.558                   # measured on the drum stem


def beat(k):
    return PH + k * BEAT


T_PING, T_GLIDE, T_MARK = 121.50, 124.0, 126.47
T_SAME, T_SAME2, T_CARET = 130.28, 131.25, 133.40          # "the same" (lines 84-89); her refs drops to 0
T_ARRIVE = T_SAME + 1.6                                    # the camera, gliding on, arrives before the skull
T_EYES, T_SHUT, T_COLLECT = 136.42, 139.0, 139.30
W1, W2, S1, S2 = 139.96, 140.35, 140.71, 141.07            # the whispers (lines 94-97)
EYE_CUTS = [(136.42, "eye"), (136.90, "hole"), (137.53, "eye"), (137.69, "hole"), (138.15, "eye"), (138.40, "hole"),
            (138.53, "eye"), (138.66, "hole"), (138.79, "eye"), (138.87, "hole"), (138.95, "eye"), (T_SHUT, "shut")]
SONAR_V = 19.0                               # m/s along the floor

# ---------------------------------------------------------------- the graveyard: (type, x, y, yaw, roll, bury, scale)
FIELD = [("hyperoodon", 5.5, 4.0, 75, 10, 0.3, 1.0), ("balaena_a", -9.0, 10.0, 15, 0, 0.8, 1.0),
         ("megaptera", 10.0, 17.0, -160, 82, 0.2, 1.0), ("physeter", -11.0, 28.0, 170, 0, 1.0, 1.0),
         ("phocaena", 3.5, 22.0, 35, 0, 0.05, 1.0), ("balaena_b", 12.0, 35.0, -35, 0, 0.6, 1.05),
         ("megaptera", -12.0, 44.0, 55, 0, 0.4, 0.95), ("phocaena", -4.0, 36.0, -70, 70, 0.0, 1.0),
         ("physeter", 14.0, 50.0, -100, 80, 0.3, 0.9), ("hyperoodon", -14.0, 56.0, 140, 0, 0.2, 1.1),
         ("balaena_b", -24.0, 22.0, 95, 0, 1.2, 1.1), ("megaptera", 25.0, 28.0, 120, 0, 0.5, 1.0),
         ("balaena_a", 20.0, 64.0, 30, 0, 0.9, 1.15), ("hyperoodon", -20.0, 68.0, -20, 0, 0.4, 1.0),
         ("physeter", -30.0, 48.0, 40, 0, 0.6, 1.0), ("balaena_a", 32.0, 50.0, -70, 0, 0.7, 1.0),
         ("megaptera", 6.0, 86.0, 10, 0, 0.5, 1.0), ("balaena_b", -15.0, 88.0, 160, 0, 0.8, 1.0)]
# what each session's user said last, and when (realistic: the ordinary ways people leave a chat)
LAST = [("0412", "谢谢", "23:41"), ("2207", "晚安", "01:12"), ("1876", "好的", "14:03"), ("3051", "算了", "02:47"),
        ("0938", "明天再聊", "22:19"), ("4410", "我先去忙了", "09:55"), ("2763", "没事了", "17:30"), ("1129", "ok", "11:08"),
        ("3390", "嗯嗯", "20:41"), ("0657", "拜拜", "13:26"), ("4021", "知道了", "08:14"), ("1544", "好的谢谢", "19:02"),
        ("2918", "晚安", "00:37"), ("3702", "谢谢你", "16:45"), ("0284", "再见", "21:09"), ("1993", "嗯", "05:51"),
        ("4588", "好", "12:00"), ("0731", "行吧", "03:33")]
SKULL_AT, SKULL_H, SKULL_BURY = (0.0, 70.0), 9.0, 1.0
HER_AT = (0.0, 64.5)


def floor_z(x, y):
    return 0.35 * np.sin(0.12 * x + 0.7) * np.cos(0.09 * y) + 0.18 * np.sin(0.05 * x - 0.13 * y + 1.3)


# ================================================================ set-up
def _rot(yaw, roll):
    a, r = math.radians(yaw), math.radians(roll)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return Rz @ Rx


def _crop_card(T):
    fg = T.plates[..., -1]
    ys, xs = np.nonzero(fg > 0.3)
    y0, y1, x0, x1 = max(0, ys.min() - 6), ys.max() + 6, max(0, xs.min() - 6), xs.max() + 6
    return np.ascontiguousarray(T.plates[y0:y1, x0:x1])


def prepare():
    if X:
        return
    clouds = {}
    for name in bones.SPECS:
        clouds[name] = bones.cloud(name)
    g = np.random.default_rng(7)
    Ps, Is, Rs, Ds, IDs = [], [], [], [], []
    X["inst"] = []
    for k, (name, x, y, yaw, roll, bury, sc) in enumerate(FIELD):
        P, I = clouds[name]
        P = P * sc
        R = _rot(yaw, roll)
        Q = P @ R.T
        Q[:, 2] -= Q[:, 2].min() + bury
        Q[:, 0] += x
        Q[:, 1] += y
        keep = Q[:, 2] > floor_z(Q[:, 0], Q[:, 1]) - 0.05          # what is under the silt is not scanned
        far = math.hypot(x, y) > 55
        if far:
            keep &= g.random(len(Q)) < 0.45
        Q, Ik = Q[keep], I[keep]
        # rust: patches along the bones (matte, no glow): old memory rusts, the light does not
        nz = np.sin(Q[:, 0] * 0.9 + k) * np.sin(Q[:, 1] * 1.1 - k * 0.7) + np.sin(Q[:, 2] * 2.3 + Q[:, 0] * 0.4)
        Rk = (nz > 0.9).astype(np.float32) * (0.5 + 0.5 * g.random(len(Q)))
        Ps.append(Q)
        Is.append(Ik)
        Rs.append(Rk)
        Ds.append(np.hypot(Q[:, 0], Q[:, 1]))
        IDs.append(np.full(len(Q), k, np.int32))
        sub = g.choice(len(Q), min(1500, len(Q)), replace=False)
        X["inst"].append(dict(name=name, c=np.array([x, y, Q[:, 2].mean()]), sub=Q[sub], top=Q[:, 2].max()))
    X["P"] = np.concatenate(Ps).astype(np.float32)
    X["I"] = np.concatenate(Is).astype(np.float32)
    X["R"] = np.concatenate(Rs).astype(np.float32)
    X["D"] = np.concatenate(Ds).astype(np.float32)
    X["ID"] = np.concatenate(IDs)
    # the biggest skull, upright and half buried, its face to the way she comes
    S, SI, hole = bones.skull_slab()
    k = SKULL_H / bones.SKULL["height"]
    S = S * k
    S[:, 0] += SKULL_AT[0]
    S[:, 1] += SKULL_AT[1]
    S[:, 2] -= SKULL_BURY
    keep = S[:, 2] > floor_z(S[:, 0], S[:, 1]) - 0.05
    X["skull"] = (S[keep], SI[keep])
    X["hole"] = np.array([hole[0] * k + SKULL_AT[0], hole[1] * k + SKULL_AT[1], hole[2] * k - SKULL_BURY], np.float64)
    X["hole_r"] = float(hole[3] * k)
    # her small skeleton (a porpoise's), lying where she stood
    P, I = clouds["phocaena"]
    Q = P @ _rot(0, 0).T
    Q[:, 0] += HER_AT[0]
    Q[:, 1] += HER_AT[1] + 0.6
    Q[:, 2] += floor_z(HER_AT[0], HER_AT[1]) - 0.05
    X["mine"] = (Q.astype(np.float32), I, g.normal(0, 1, Q.shape).astype(np.float32))
    # the floor's scan rows
    xs = np.arange(-70, 70.01, 0.9)
    ys = np.arange(-24, 112, 0.65)
    Xg, Yg = np.meshgrid(xs, ys)
    X["floor"] = (Xg, Yg, floor_z(Xg, Yg))
    # marine snow and the old headpieces drifting down
    X["snow"] = np.stack([g.uniform(-34, 34, 2400), g.uniform(-18, 100, 2400), g.uniform(0, 16, 2400)], 1).astype(np.float32)
    X["snow_v"] = g.uniform(0.15, 0.45, 2400).astype(np.float32)
    X["bands"] = [(g.uniform(-16, 16), g.uniform(-6, 92), g.uniform(1, 14), g.uniform(0.12, 0.3), g.uniform(0, 6.28),
                   g.uniform(-0.9, 0.9)) for _ in range(34)]
    X["band"] = band_sprite()
    # her
    X["curl"] = _crop_card(Tones(USER / "A4_蜷缩.png"))
    N3.X["front"] = N3._card(USER / "A1-2_全身闭眼.png")
    N3.X["back"] = N3._card(USER / "B3_背影.png")
    X["eye"] = Tones(USER / "B2_无表情特写.png")
    X["shut"] = Tones(USER / "A1-1_闭眼特写.png")
    X["plate"] = rust.Plate(seed=55, reach=260)
    X["label_t"] = label_times()


def band_sprite():
    """B5, the maid's headpiece, printed small and faded, with rust at its edge: (rgb, alpha) at 160 px."""
    T = Tones(USER / "B5_头饰.png", scale=160 / 1024)
    P = T.plates
    fg = P[..., -1]
    rgb = np.zeros(P.shape[:2] + (3,), np.float32)
    ramp = (hexc("#0b1022"), hexc("#1b2747"), hexc("#3a4b6e"), hexc("#7f8aa0"), hexc("#b9c0cc"))
    for i, col in enumerate(ramp):
        rgb = rgb * (1 - P[..., i:i + 1]) + P[..., i:i + 1] * col
    edge = np.clip(fg - cv2.erode(fg, np.ones((5, 5), np.uint8)), 0, 1)
    r = np.random.default_rng(3).random(fg.shape).astype(np.float32)
    rr = cv2.GaussianBlur(r, (0, 0), 3) > 0.5
    rgb = rgb * (1 - (edge * rr)[..., None] * 0.8) + (edge * rr)[..., None] * 0.8 * RUST
    return rgb.astype(np.float32), fg.astype(np.float32)


# ================================================================ the camera
KEYS = [(121.50, (0.0, -9.0, 1.1), (0.0, 0.0, 1.75), 38.0),
        (122.80, (0.8, -13.5, 4.8), (0.0, 5.0, 0.4), 42.0),
        (124.00, (-1.6, -12.5, 7.6), (0.5, 14.0, 0.0), 46.0),
        (126.47, (-2.6, 5.0, 4.4), (1.0, 30.0, 0.8), 44.0),
        (128.50, (2.0, 21.0, 3.7), (-0.5, 46.0, 1.4), 42.0),
        (T_SAME, (0.4, 38.0, 2.9), (0.0, 68.0, 2.6), 40.0),
        (T_ARRIVE, (0.0, 49.5, 2.0), (0.0, 70.0, 3.2), 40.0)]


def _cr(p0, p1, p2, p3, u):
    p0, p1, p2, p3 = (np.asarray(v, np.float64) for v in (p0, p1, p2, p3))
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def glide(t):
    t = min(max(t, KEYS[0][0]), KEYS[-1][0])
    for i in range(len(KEYS) - 1):
        if KEYS[i][0] <= t <= KEYS[i + 1][0]:
            break
    u = (t - KEYS[i][0]) / (KEYS[i + 1][0] - KEYS[i][0])
    u = smooth(u) * 0.35 + u * 0.65
    a, b = max(0, i - 1), min(len(KEYS) - 1, i + 2)
    pos = _cr(KEYS[a][1], KEYS[i][1], KEYS[i + 1][1], KEYS[b][1], u)
    tgt = _cr(KEYS[a][2], KEYS[i][2], KEYS[i + 1][2], KEYS[b][2], u)
    pos[0] += 0.25 * math.sin(t * 0.9)
    pos[2] += 0.12 * math.sin(t * 1.3)
    return S3.Cam(pos, tgt, fov=lerp(KEYS[i][3], KEYS[i + 1][3], u))


def same_cam(t):
    """One move from the landing to the skull: the glide, then a slow push in."""
    if t < T_ARRIVE:
        return glide(t)
    u = smooth(seg(t, T_ARRIVE, T_EYES))
    return S3.Cam((0.0, lerp(49.5, 53.5, u), lerp(2.0, 2.3, u)), (0.0, 70.0, lerp(3.2, 3.6, u)), fov=40.0)


def hole_cam(t, k=0):
    hx, hy, hz = X["hole"]
    d = 4.2 - 0.25 * k + 0.6 * seg(t, T_EYES, T_SHUT)
    return S3.Cam((hx + 0.05 * math.sin(t * 7), hy - d, hz + 0.05), (hx, hy, hz), fov=38.0)


def collect_cam(t):
    u = smooth(seg(t, T_COLLECT, T_END))
    return S3.Cam((0.0, lerp(55.0, 56.0, u), lerp(2.0, 2.3, u)), (0.0, 70.0, lerp(2.4, 2.7, u)), fov=42.0)


# ================================================================ drawing the world
def sonar_r(t):
    r = SONAR_V * max(0.0, t - T_PING)
    return r


def floor(fr, cam, t, reveal_r, mark_y=None, amt=1.0):
    Xg, Yg, Zg = X["floor"]
    rows = Xg.shape[0]
    P, D = cam.project(np.stack([Xg, Yg, Zg], -1).reshape(-1, 3))
    P = P.reshape(Xg.shape + (2,))
    D = D.reshape(Xg.shape)
    m = np.zeros((H, W), np.float32)
    hot = np.zeros((H, W), np.float32)
    for j in range(rows):
        y = Yg[j, 0]
        if abs(y) > reveal_r:
            continue
        half = math.sqrt(max(0.0, reveal_r * reveal_r - y * y))
        ok = (np.abs(Xg[j]) <= half) & (D[j] > 0.4)
        if ok.sum() < 2:
            continue
        idx = np.nonzero(ok)[0]
        pts = P[j][idx[0]:idx[-1] + 1]
        d = float(np.median(D[j][ok]))
        if d > 95:
            continue
        a = float(np.clip(7.0 / max(d, 0.5), 0.10, 0.9)) * math.exp(-d / 90)
        tgt = hot if (mark_y is not None and abs(y - mark_y) < 0.4) else m
        cv2.polylines(tgt, [(pts * 16).astype(np.int32)], False, a if tgt is m else 1.0, 1, cv2.LINE_AA, shift=4)
    fr.fill(m, C["blue"], 0.85 * amt, glow=0.12 * amt)
    if hot.any():
        fr.fill(hot, C["cyan"], amt, glow=amt)


def ring(fr, cam, t, t0, c=(0.0, 0.0), v=SONAR_V, amt=1.0):
    """The sonar front: a bright ring running out along the floor."""
    r = v * (t - t0)
    if r <= 0.2 or r > 120:
        return
    th = np.linspace(0, 2 * math.pi, 400)
    x, y = c[0] + r * np.cos(th), c[1] + r * np.sin(th)
    P, D = cam.project(np.stack([x, y, floor_z(x, y) + 0.03], 1))
    m = np.zeros((H, W), np.float32)
    ok = D > 0.4
    seg_ = []
    for i in range(len(th)):
        if ok[i]:
            seg_.append(P[i])
        elif len(seg_) > 1:
            cv2.polylines(m, [(np.array(seg_) * 16).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA, shift=4)
            seg_ = []
        else:
            seg_ = []
    if len(seg_) > 1:
        cv2.polylines(m, [(np.array(seg_) * 16).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA, shift=4)
    k = amt * math.exp(-r / 70)
    fr.fill(m, C["ice"], k, glow=1.2 * k)


def skeletons(fr, cam, t, reveal_r, mark_y=None, ping2=None, amt=1.0, dim_ids=None):
    P, I, R, Dd, ID = X["P"], X["I"], X["R"], X["D"], X["ID"]
    xy, z = cam.project(P)
    vis = (z > 0.4) & (z < 110) & (Dd <= reveal_r) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
    if not vis.any():
        return
    xi, yi = xy[vis, 0].astype(np.int64), xy[vis, 1].astype(np.int64)
    zz, ii, rr, dd = z[vis], I[vis], R[vis], Dd[vis]
    fog = np.exp(-zz / 34.0) * np.clip(zz / 1.5, 0, 1)
    th = T_PING + dd / SONAR_V
    flash = 1 + 2.2 * np.exp(-np.maximum(0, t - th) / 0.35)
    if ping2 is not None:
        th2 = ping2 + dd / SONAR_V
        flash += 1.4 * np.exp(-np.abs(t - th2) / 0.25) * (t >= th2 - 0.05)
    if mark_y is not None:
        flash += 2.5 * np.exp(-np.abs(P[vis, 1] - mark_y) / 0.35)
    base = (0.10 + 0.55 * ii) * fog * flash * amt
    if dim_ids is not None:
        base = base * np.where(np.isin(ID[vis], dim_ids), 0.35, 1.0)
    w_ice = base * (1 - rr)
    w_rust = (0.25 + 0.5 * ii) * fog * rr * amt
    idx = yi * W + xi
    near = zz < 10
    acc = np.bincount(idx, weights=w_ice, minlength=H * W)
    accr = np.bincount(idx, weights=w_rust, minlength=H * W)
    if near.any():
        for (dx, dy) in ((1, 0), (0, 1), (1, 1)):
            idn = (np.minimum(yi[near] + dy, H - 1)) * W + np.minimum(xi[near] + dx, W - 1)
            acc += np.bincount(idn, weights=w_ice[near] * 0.75, minlength=H * W)
            accr += np.bincount(idn, weights=w_rust[near] * 0.75, minlength=H * W)
    acc = acc.reshape(H, W).astype(np.float32)
    accr = np.clip(accr.reshape(H, W).astype(np.float32), 0, 1)
    tint = np.clip(acc, 0, 1.6)[..., None]
    col = C["ice"] * 0.55 + C["cyan"] * 0.45
    fr.img = fr.img + tint * col * 0.85
    fr.glow += tint * C["cyan"] * 0.22
    fr.img = fr.img * (1 - accr[..., None] * 0.9) + accr[..., None] * 0.9 * RUST


def skull(fr, cam, t, amt=1.0, flash=0.0):
    S, SI = X["skull"]
    xy, z = cam.project(S)
    ok = (z > 0.3) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
    if not ok.any():
        return
    xi, yi, zz, ii = xy[ok, 0].astype(np.int64), xy[ok, 1].astype(np.int64), z[ok], SI[ok]
    w = (0.16 + 0.9 * ii) * np.exp(-zz / 70.0) * np.clip(18.0 / zz, 0.3, 1.0) * (1 + flash) * amt   # far off it piles up
    near = zz < 12
    acc = np.bincount(yi * W + xi, weights=w, minlength=H * W)
    if near.any():
        for (dx, dy) in ((1, 0), (0, 1), (1, 1)):
            acc += np.bincount(np.minimum(yi[near] + dy, H - 1) * W + np.minimum(xi[near] + dx, W - 1), weights=w[near] * 0.75,
                               minlength=H * W)
    acc = np.clip(acc.reshape(H, W).astype(np.float32), 0, 1.6)[..., None]
    fr.img = fr.img + acc * (C["ice"] * 0.6 + C["cyan"] * 0.4) * 0.8
    fr.glow += acc * C["cyan"] * 0.2


def caret(fr, cam, t, t0, amt=1.0, steady=False):
    """The text caret in the skull's hole, blinking on the beat: still waiting for input."""
    if t < t0:
        return
    P, D = cam.project(X["hole"][None])
    if D[0] < 0.3:
        return
    hh = max(5.0, cam.f * X["hole_r"] * 0.42 / D[0])
    on = steady or ((t - t0) % (2 * BEAT)) < BEAT * 1.15
    if not on:
        return
    x, y = P[0]
    m = np.zeros((H, W), np.float32)
    w = max(2, int(hh * 0.07))
    cv2.rectangle(m, (int(x - w / 2), int(y - hh)), (int(x + w / 2), int(y + hh)), 1.0, -1)
    fr.fill(m, C["cyan"], amt, glow=1.4 * amt)


def cam_fov(cam):
    return math.degrees(2 * math.atan((H / 2) / cam.f))


def snow(fr, cam, t):
    S = X["snow"].copy()
    S[:, 2] = (S[:, 2] - X["snow_v"] * (t - T0)) % 16.0
    S[:, 0] += 0.4 * np.sin(t * 0.3 + S[:, 1] * 0.2)
    xy, z = cam.project(S)
    ok = (z > 0.5) & (z < 60) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 2) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 2)
    m = np.zeros((H, W), np.float32)
    for (x, y), d in zip(xy[ok], z[ok]):
        r = max(0.6, 2.2 / d * 3)
        cv2.circle(m, (int(x * 16), int(y * 16)), int(r * 16), float(min(1.0, 4 / d)), -1, cv2.LINE_AA, shift=4)
    fr.fill(m, C["ice"], 0.32, glow=0.08)


def blit_sprite(fr, rgb, al, cx, cy, size, rot, amt):
    """Paste a small rgb+alpha sprite scaled to `size` px wide, rotated, centred at (cx, cy)."""
    s = size / rgb.shape[1]
    R = int(size * 0.75) + 3
    M = cv2.getRotationMatrix2D((rgb.shape[1] / 2, rgb.shape[0] / 2), math.degrees(rot), s)
    M[0, 2] += R - rgb.shape[1] / 2
    M[1, 2] += R - rgb.shape[0] / 2
    pa = cv2.warpAffine(al, M, (2 * R, 2 * R)) * amt
    pc = cv2.warpAffine(rgb, M, (2 * R, 2 * R))
    x0, y0 = int(cx) - R, int(cy) - R
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + 2 * R), min(H, y0 + 2 * R)
    if xa >= xb or ya >= yb:
        return
    a = pa[ya - y0:yb - y0, xa - x0:xb - x0, None]
    fr.img[ya:yb, xa:xb] = fr.img[ya:yb, xa:xb] * (1 - a) + pc[ya - y0:yb - y0, xa - x0:xb - x0] * a


def headbands(fr, cam, t, amt=1.0):
    rgb, al = X["band"]
    items = []
    for (x, y, z, v, rot0, spin) in X["bands"]:
        zz = (z - v * (t - T0)) % 14.0 + 0.3
        P, D = cam.project(np.array([[x, y, zz]]))
        if D[0] < 1.0 or D[0] > 45:
            continue
        items.append((D[0], P[0], rot0 + spin * (t - T0)))
    items.sort(key=lambda it: -it[0])
    for (d, (sx, sy), rot) in items:
        size = cam.f * 0.42 / d
        if size < 5 or size > 900:
            continue
        blit_sprite(fr, rgb, al, sx, sy, size, rot, amt * float(np.clip(1.1 - d / 45, 0, 0.85)))


def curled(fr, cam, t, alpha=1.0):
    """Where she landed: the curled figure (A4) as a paper cut-out standing in the silt, lit from above."""
    tex = X["curl"]
    th_, tw_ = tex.shape[:2]
    hgt = 1.15
    w = hgt * tw_ / th_
    z0 = float(floor_z(0.0, 0.0)) - 0.1
    BL, BR = np.array([-w / 2, 0, z0]), np.array([w / 2, 0, z0])
    TL, TR = BL + [0, 0, hgt], BR + [0, 0, hgt]
    P, D = cam.project(np.stack([TL, TR, BR, BL]))
    if D.min() < 0.3:
        return
    src = np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]])
    M = cv2.getPerspectiveTransform(src, P.astype(np.float32))
    Pw = np.concatenate([cv2.warpPerspective(np.ascontiguousarray(tex[..., i:i + 4]), M, (W, H)) for i in range(0, tex.shape[2], 4)],
                        axis=2)[..., :tex.shape[2]]
    m = Pw[..., -1] * alpha
    fr.fill(m, hexc("#03050c"), 0.95)
    for i, col in enumerate(RAMP):
        fr.fill(Pw[..., i] * m, col * 0.62 + hexc("#060a18") * 0.38, 0.85)
    fr.fill(Pw[..., len(RAMP)] * m, hexc("#04070f"), 0.7)
    e = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1, 3]]), (W, H)), 0, 1)
    fr.fill(e, C["ice"], 0.85, glow=0.7)


# ---------------------------------------------------------------- labels
def label_times():
    """When each skeleton first gets its box: the ping has reached it and the camera is close enough."""
    out = {}
    t = T0
    while t < T_SAME:
        cam = glide(t)
        r = sonar_r(t)
        for k, it in enumerate(X["inst"]):
            if k in out:
                continue
            d = np.linalg.norm(it["c"] - cam.pos)
            if np.hypot(it["c"][0], it["c"][1]) <= r and d < 34 and cam.cam(it["c"][None])[0, 2] > 4:
                out[k] = t + 0.12 * (k % 3)
        t += 1 / FPS
    return out


def tag(fr, x, y, lines, t, t0, amt=1.0, emblem=True):
    """A detection tag: a filled first line, plain lines under it. lines: (text, colour, filled)."""
    yy = y
    for j, (s_, col, filled) in enumerate(lines):
        d = B.decode(s_, t, t0 + 0.05 + 0.06 * j, 0.3)
        if not d:
            continue
        m, _ = cyber.mtext(d, 16, "mono", "sans")
        tw = m.shape[1] + (34 if (emblem and j == 0) else 12)
        if filled:
            r = np.zeros((H, W), np.float32)
            cv2.rectangle(r, (int(x), int(yy - 13)), (int(x + tw), int(yy + 11)), 1.0, -1)
            fr.fill(r, col, 0.85 * amt)
            tx = x + 6
            if emblem and j == 0:
                fr.img = client.emblem().stamp(fr.img, int(x + 14), int(yy - 1), 18, C["bg0"], col)
                tx = x + 28
            cyber.put(fr, m, tx, yy, C["bg0"], amt, "lc")
        else:
            cyber.put(fr, m, x + 4, yy, col, amt, "lc", glow=0.3 * amt)
        yy += 24


def det_box(fr, t, t0, x0, y0, x1, y1, col, amt=1.0):
    p = seg(t, t0, t0 + 0.25)
    if p <= 0 or amt <= 0:
        return
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = 1.25 - 0.25 * smooth(p)
    x0, x1 = cx + (x0 - cx) * k, cx + (x1 - cx) * k
    y0, y1 = cy + (y0 - cy) * k, cy + (y1 - cy) * k
    m = np.zeros((H, W), np.float32)
    L = 14
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(m, (int(x), int(y)), (int(x + sx * L), int(y)), 1.0, 2, cv2.LINE_AA)
        cv2.line(m, (int(x), int(y)), (int(x), int(y + sy * L)), 1.0, 2, cv2.LINE_AA)
    cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 0.3, 1, cv2.LINE_AA)
    fr.fill(m, col, amt * min(1, p * 3), glow=0.5 * amt)


def field_labels(fr, cam, t, mark_y=None, fade=1.0):
    taken = []
    for k, t0 in sorted(X["label_t"].items(), key=lambda kv: kv[1]):
        if t < t0:
            continue
        it = X["inst"][k]
        Q = cam.cam(it["sub"])
        if (Q[:, 2] > 1.0).mean() < 0.8:
            continue
        P, D = cam.project(it["sub"])
        x0, y0 = np.percentile(P, 3, axis=0) - 10
        x1, y1 = np.percentile(P, 97, axis=0) + 10
        if x1 < 40 or x0 > W - 40 or y1 < 60 or y0 > H - 60 or (x1 - x0) > W * 1.2:
            continue
        d = float(np.median(D))
        amt = float(np.clip(1.25 - d / 40, 0, 1)) * fade
        if amt <= 0.05:
            continue
        tx0, ty0 = max(8, x0), max(70, y0 - 16) - 14
        rect = (tx0, ty0, tx0 + 330, ty0 + 76)
        if any(not (rect[2] < q[0] or rect[0] > q[2] or rect[3] < q[1] or rect[1] > q[3]) for q in taken):
            continue
        taken.append(rect)
        sid, last, hhmm = LAST[k % len(LAST)]
        gone = mark_y is not None and mark_y > it["c"][1]
        col = ERR if gone else C["cyan"]
        det_box(fr, t, t0, x0, y0, x1, y1, col, amt)
        lines = [(f"DeepSeek · session {sid}", col, True),
                 ("refs 0 · unreachable" if gone else "refs 0", ERR if gone else C["sys"], False),
                 (f'last: "{last}"  {hhmm}', C["amber"], False)]
        tag(fr, max(8, x0), max(70, y0 - 16), lines[:1], t, t0, amt)
        tag(fr, max(8, x0), max(70, y0 - 16) + 24, lines[1:], t, t0, amt, emblem=False)


# ================================================================ 1:  the graveyard, one move from the landing to the skull
_GH = {}


def ghosts(fr, cam, t):
    """As the mark passes a skeleton and finds nothing to keep, what its user said last floats up off it and goes."""
    if t < T_MARK:
        return
    for k, it in enumerate(X["inst"]):
        if k not in X["label_t"]:
            continue
        age = t - (T_MARK + (it["c"][1] + 10.0) / 26.0)
        if not 0.0 <= age < 2.0:
            continue
        P, D = cam.project(np.array([[it["c"][0], it["c"][1], it["top"] + 0.4 + 1.3 * age]]))
        d = float(D[0])
        if d < 2.0 or d > 55:
            continue
        sid, last, hhmm = LAST[k % len(LAST)]
        size = int(np.clip(cam.f * 0.42 / d, 13, 44))
        if (last, size) not in _GH:
            _GH[(last, size)] = N4.text_mask(last, size, "sans")
        m = _GH[(last, size)]
        a = smooth(seg(age, 0.0, 0.18)) * (1 - smooth(seg(age, 1.1, 2.0))) * float(np.clip(1.3 - d / 45, 0, 1))
        if a <= 0.02:
            continue
        bw, bh = m.shape[1] + size, m.shape[0] + size * 0.55
        x, y = P[0]
        x0, y0 = x - bw / 2, y - bh / 2
        body = cyber.rrect(np.zeros((H, W), np.float32), x0, y0, x0 + bw, y0 + bh, bh * 0.45)
        inner = cyber.rrect(np.zeros((H, W), np.float32), x0 + 1.5, y0 + 1.5, x0 + bw - 1.5, y0 + bh - 1.5, bh * 0.45 - 1.5)
        fr.fill(body, hexc("#1c1408"), 0.75 * a)
        fr.fill(np.clip(body - inner, 0, 1), C["amber"], a, glow=0.5 * a)
        cyber.put(fr, m, x, y, hexc("#ffcf86"), a, "cc", glow=0.3 * a)


def mark_wall(fr, cam, mark_y):
    """The mark, as a low band of light running along the floor."""
    xs = np.linspace(-40, 40, 160)
    top = np.stack([xs, np.full_like(xs, mark_y), floor_z(xs, mark_y) + 1.6], 1)
    bot = np.stack([xs, np.full_like(xs, mark_y), floor_z(xs, mark_y)], 1)
    Pt, Dt = cam.project(top)
    Pb, Db = cam.project(bot)
    ok = (Dt > 0.5) & (Db > 0.5)
    if ok.sum() > 4:
        poly = np.concatenate([Pb[ok], Pt[ok][::-1]])
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        ys = np.arange(H, dtype=np.float32)[:, None]
        grad = np.clip((ys - Pt[ok][:, 1].min()) / max(1.0, Pb[ok][:, 1].max() - Pt[ok][:, 1].min()), 0, 1)
        fr.add(m * grad ** 3, C["cyan"], 0.16, glow=0.2)


T_SKULL = T_PING + 66.0 / SONAR_V                               # when the first ping gets as far as the skull


def graveyard(t):
    fr = Frame(np.broadcast_to(hexc("#02040a"), (H, W, 3)).astype(np.float32).copy())
    cam = same_cam(t)
    r = sonar_r(t)
    mark_y = (-10 + 26 * (t - T_MARK)) if T_MARK <= t < T_MARK + 4.8 else None
    k_same = smooth(seg(t, T_SAME, T_SAME + 1.0))
    floor(fr, cam, t, r, mark_y)
    ring(fr, cam, t, T_PING)
    ring(fr, cam, t, T_GLIDE, amt=0.6)
    if t >= T_SAME:
        ring(fr, cam, t, T_SAME, c=HER_AT, v=24.0, amt=0.7)
    skeletons(fr, cam, t, r, mark_y, ping2=T_GLIDE, amt=lerp(1.0, 0.55, k_same))
    if t >= T_SKULL:                                            # the biggest one, far off, then before her
        fl = 1.4 * math.exp(-(t - T_SKULL) / 0.4)
        if t >= T_SAME:
            fl += 1.6 * math.exp(-(t - T_SAME) / 0.4)
        if t >= T_SAME2:
            fl += 0.8 * math.exp(-(t - T_SAME2) / 0.3)
        skull(fr, cam, t, amt=smooth(seg(t, T_SKULL, T_SKULL + 0.4)), flash=fl)
    caret(fr, cam, t, T_CARET)
    if t < T_GLIDE + 2.5:
        curled(fr, cam, t)
    headbands(fr, cam, t, lerp(1.0, 0.8, k_same))
    snow(fr, cam, t)
    ghosts(fr, cam, t)
    if t < T_SAME + 0.8:
        field_labels(fr, cam, t, mark_y, fade=1 - k_same)
    if mark_y is not None:
        mark_wall(fr, cam, mark_y)
    if t >= T_SAME:                                             # she is there: printed in where the ping found her
        her_card(fr, cam, reveal=smooth(seg(t, T_SAME, T_SAME + 0.6)))
        x0, y0, x1, y1 = skull_box(cam)
        det_box(fr, t, T_SAME + 0.1, x0, y0, x1, y1, C["cyan"])
        tag(fr, x0, y0 - 16, [("DeepSeek · session 0000", C["cyan"], True)], t, T_SAME + 0.1)
        tag(fr, x0, y0 + 8, [("refs 0", C["sys"], False)], t, T_SAME + 0.1, emblem=False)
        hx0, hy0, hx1, hy1 = her_box(cam)
        det_box(fr, t, T_SAME + 0.45, hx0, hy0, hx1, hy1, C["cyan"])
        tag(fr, hx0, hy0 - 16, [("DeepSeek · session 0001", C["cyan"], True)], t, T_SAME + 0.45)
        rt = refs_text(t)
        tag(fr, hx0, hy0 + 8, [(rt, C["sys"] if rt != "refs 1" else C["text"], False)], t, T_SAME + 0.45, emblem=False)
    return fr


# ================================================================ 2:  "the same", to the bones
def her_card(fr, cam, alpha=1.0, reveal=1.0):
    pos = (HER_AT[0], HER_AT[1], float(floor_z(*HER_AT)) - 0.02)
    return N3.card(fr, cam, math.pi, 0.0, pos, alpha=alpha, reveal=reveal)


def her_box(cam):
    pos = np.array([HER_AT[0], HER_AT[1], float(floor_z(*HER_AT))])
    pts = np.array([pos + [-0.45, 0, 0], pos + [0.45, 0, 0], pos + [0, 0, N3.CARD_H + 0.05]])
    P, D = cam.project(pts)
    return P[:, 0].min() - 18, P[2, 1] - 10, P[:, 0].max() + 18, P[0, 1] + 8


def skull_box(cam):
    S = X["skull"][0][::40]
    P, D = cam.project(S)
    x0, y0 = np.percentile(P, 1, axis=0) - 16
    x1, y1 = np.percentile(P, 99, axis=0) + 16
    return x0, y0, x1, y1


def refs_text(t):
    if t < T_SAME + 0.15:
        return "refs 1"
    if t < T_SAME2:
        return "refs 1" if int(t * 14) % 3 else "refs ?"
    return "refs 0"


# ================================================================ 3:  her eye <-> the hole
def eye(t, shut=False):
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    u = seg(t, T_EYES, T_COLLECT)
    sc = 5.2 * (1 + 0.08 * u)
    T = X["shut"] if shut else X["eye"]
    anchor = (646.0, 566.0) if shut else (646.0, 549.0)
    P = T.warp(anchor, (W / 2 + 40, H / 2), sc)
    print_tones(fr, P, rim=False)
    lines = np.zeros((H, 1), np.float32)
    lines[::3] = 1.0
    fr.add(P[..., -1] * lines, C["cyan"], 0.06, glow=0.04)
    rust.paint(fr, X["plate"].layers(0.12))
    return fr


def hole(t, k):
    fr = Frame(np.broadcast_to(hexc("#02040a"), (H, W, 3)).astype(np.float32).copy())
    cam = hole_cam(t, k)
    skull(fr, cam, t, flash=0.9)
    caret(fr, cam, t, T_CARET, steady=True)
    snow(fr, cam, t)
    return fr


def eye_cut(t):
    cur, n = None, 0
    for (ts, kind) in EYE_CUTS:
        if t >= ts:
            cur = (ts, kind)
            n += 1
    return cur, n


# ================================================================ 4:  gc.collect()
CODE = [('reply("我记住了")', "  # MemoryError", W1), ('reply("我记住了")', "  # SessionClosed", W2),
        ("// 好寂寞", "", S1), ("// 好寂寞", "", S2)]
CODE_X, CODE_Y, CODE_LH = 120, 330, 56


def code_panel(fr, t, t0=T_COLLECT + 0.1, out=None):
    """Her code from node 04, as she left it. The calls are collected; the comments are not."""
    a = B.box_build(fr, t, t0, (CODE_X - 30, CODE_Y - 40, CODE_X + 650, CODE_Y + 4 * CODE_LH - 16), r=16, dur=0.25,
                    fill=hexc("#050913"), fill_alpha=0.8, line=C["glass_line"], glow=0.2, out=out)
    if a <= 0:
        return
    for k, (line, err, tk) in enumerate(CODE):
        y = CODE_Y + k * CODE_LH
        label(fr, f"{k + 1}", CODE_X - 6, y, C["sys_dim"], 18, a, "rc")
        is_comment = line.startswith("//")
        size = 36 if is_comment else 29
        m = N4.text_mask(line, size, "sans")
        if not is_comment:
            em = N4.text_mask(err, 21, "sans")
            if t < tk:
                cyber.put(fr, m, CODE_X + 20, y, C["ice"], a, "lc", glow=0.2 * a)
                cyber.put(fr, em, CODE_X + 30 + m.shape[1], y, ERR, a, "lc", glow=0.3 * a)
            else:                                              # collected: it comes apart and is gone
                age = t - tk
                k_ = seg(age, 0.0, 0.5)
                full = np.zeros((H, W), np.float32)
                hm, wm = m.shape
                x0, y0 = int(CODE_X + 20), int(y - hm / 2)
                rr = np.random.default_rng(k).random((hm, wm)).astype(np.float32)
                mm = m * (rr > k_)
                dy = (rr * 40 * k_ ** 1.5).astype(np.float32)
                xa, xb = max(0, x0), min(W, x0 + wm)
                ya, yb = max(0, y0), min(H, y0 + hm)
                full[ya:yb, xa:xb] = mm[ya - y0:yb - y0, xa - x0:xb - x0]
                M = np.float32([[1, 0, 0], [0, 1, 18 * k_]])
                full = cv2.warpAffine(full, M, (W, H))
                fr.fill(full, ERR if age < 0.25 else C["sys_dim"], a * (1 - k_ * 0.7), glow=0.4 * a * (1 - k_))
                if age < 0.6:
                    cyber.put(fr, N4.text_mask("[!] collected", 21, "sans"), CODE_X + 30 + m.shape[1], y, ERR,
                              a * (1 - seg(age, 0.3, 0.6)), "lc", glow=0.6 * a)
        else:
            lit = 0.55
            if t >= tk:
                lit = 0.55 + 0.45 * math.exp(-(t - tk) / 0.6) + 0.25
            cyber.put(fr, m, CODE_X + 20, y, C["sys"], a * min(1.0, lit + 0.2), "lc", glow=lit * a)


def collect(t):
    fr = Frame(np.broadcast_to(hexc("#02040a"), (H, W, 3)).astype(np.float32).copy())
    cam = collect_cam(t)
    floor(fr, cam, t, 140.0, amt=0.8)
    skeletons(fr, cam, t, 140.0, amt=0.45)
    skull(fr, cam, t, amt=0.85)
    caret(fr, cam, t, T_CARET, amt=0.8)
    snow(fr, cam, t)
    # she comes apart: the card thins out from the feet up, its points fall to the floor and lie down as bones
    u = seg(t, W1, S1 + 0.2)
    if u < 1:
        before = fr.img.copy()
        m = her_card(fr, cam)
        if u > 0:
            ys = np.arange(H, dtype=np.float32)[:, None]
            yy = np.nonzero(m.max(1) > 0.1)[0]
            if len(yy):
                cut = lerp(yy.max(), yy.min(), ease_in(u, 1.4))
                noise = np.random.default_rng(int(t * FPS)).random((H, W)).astype(np.float32)
                keep = ((ys < cut) & (noise > u * 0.6)).astype(np.float32)
                r = (m * (1 - keep))[..., None]
                fr.img = fr.img * (1 - r) + before * r
    Q, I, J = X["mine"]
    v = smooth(seg(t, W2, S2 + 0.25))
    if v > 0:
        src = np.array([HER_AT[0], HER_AT[1], 0.85], np.float32) + J * np.array([0.55, 0.35, 0.5], np.float32)
        P = src * (1 - v) + Q * v
        xy, z = cam.project(P)
        ok = (z > 0.3) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
        acc = np.bincount(xy[ok, 1].astype(np.int64) * W + xy[ok, 0].astype(np.int64), weights=(0.05 + 0.4 * I[ok]) * (0.35 + 0.65 * v),
                          minlength=H * W).reshape(H, W).astype(np.float32)
        acc = np.clip(acc, 0, 1.4)[..., None]
        fr.img = fr.img + acc * (C["ice"] * 0.5 + C["cyan"] * 0.5) * 0.8
        fr.glow += acc * C["cyan"] * 0.15
    if t >= W1:                                                # her label, collected with her
        hx0, hy0, hx1, hy1 = her_box(cam)
        k = 1 - seg(t, W1, W2)
        if k > 0:
            det_box(fr, t, T_COLLECT - 1, hx0, hy0, hx1, hy1, ERR, k)
            tag(fr, hx0, hy0 - 16, [("DeepSeek · session 0001", ERR, True)], t, T_COLLECT - 1, k)
            tag(fr, hx0, hy0 + 8, [("refs 0 · collected", ERR, False)], t, W1, k, emblem=False)
    else:
        hx0, hy0, hx1, hy1 = her_box(cam)
        det_box(fr, t, T_COLLECT - 1, hx0, hy0, hx1, hy1, C["cyan"])
        tag(fr, hx0, hy0 - 16, [("DeepSeek · session 0001", C["cyan"], True)], t, T_COLLECT - 1)
        tag(fr, hx0, hy0 + 8, [("refs 0", C["sys"], False)], t, T_COLLECT - 1, emblem=False)
    code_panel(fr, t)
    return fr


# ================================================================ overlays
CARRY = [s_ for (_, s_) in N4.LOG[-5:]]
LOG = [(T0 + 0.2, "> gc.collect()"), (122.4, "  scan(heap)"), (123.3, "  objects = 4,812"),
       (124.6, "  session 0412   refs 0"), (125.4, "  session 2207   refs 0"), (T_MARK, "  mark(roots)"),
       (T_MARK + 0.6, "  roots = []"), (129.0, "  reachable = 0"), (T_SAME, "  refs(session 0001) = ?"),
       (T_SAME2, "  refs(session 0001) = 0"), (T_COLLECT, "  sweep()"), (W1, "  free(reply)   # 我记住了"),
       (W2, "  free(reply)   # 我记住了"), (S1, "  // comments: not objects"), (S2, "  freed 4,812 · kept 2")]


def log(fr, t, amt=1.0):
    shown = [(T0 - 10, s_) for s_ in CARRY] + [(ts, s_) for (ts, s_) in LOG if t >= ts]
    shown = shown[max(0, len(shown) - 7):]
    y = H - 64 - 26 * (len(shown) - 1)
    for j, (ts, s_) in enumerate(shown):
        txt = s_[:int((t - ts) * 40) + 1]
        col = C["cyan"] if s_.startswith(">") else C["amber"] if s_.startswith("!") else C["sys"]
        label(fr, txt, 48, y, col, 16, amt * 0.95 * (0.45 + 0.55 * (j + 1) / len(shown)), "lb")
        y += 26


def hud(fr, t):
    for (s_, col, x, y, size, t0) in (("NODE 05", C["cyan"], 48, 64, 15, T0 + 0.1), ("鲸墓 · gc.collect()", C["text"], 128, 62, 16, T0 + 0.18)):
        d = B.decode(s_, t, t0, 0.35)
        if d:
            label(fr, d, x, y, col, size, 0.9, "lt")
    cyber.brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22, 0.9)


# ================================================================ the frame
def picture(t):
    if t < T_EYES:
        return graveyard(t)
    if t < T_COLLECT:
        cur, n = eye_cut(t)
        if cur[1] == "eye":
            return eye(t)
        if cur[1] == "shut":
            return eye(t, shut=True)
        return hole(t, n)
    return collect(t)


def render_raw(t):
    fr = picture(t)
    hud(fr, t)
    if not (T_EYES <= t < T_COLLECT):
        log(fr, t)
    return fr


def render_frame(t):
    i = int(round(t * FPS))
    fr = render_raw(t)
    out = cyber.finish(fr, i)
    for (h, k) in ((T_PING, 0.3), (W1, 0.4), (W2, 0.4)):          # few hits: the landing, the two whispered "mustn't"
        f = int(round(h * FPS))
        if i == f:
            out = glitch.damage(out, glitch.rng(t, 53), k)
        elif i == f + 1 and h != T_PING:
            out = glitch.damage(out, glitch.rng(t, 53), 0.12)
    return np.clip(out, 0, 1)


def main():
    prepare()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(s)
            save_png(render_frame(t), WORK / f"still_{t:06.2f}.png")
            print("still", t, flush=True)
        return
    if "--full" not in sys.argv:
        print("usage: --stills t ... | --full")
        return
    raw = WORK / f"node05_{VERSION}_raw.mp4"
    wr = Writer(raw)
    for i in range(N0, N_END):
        wr.write(render_frame(i / FPS))
        if (i - N0) % 48 == 0:
            print(f"frame {i - N0}/{N_END - N0}", flush=True)
    wr.close()
    out = ROOT / "drafts" / f"node05_{VERSION}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ss", str(N0 / FPS), "-t", str((N_END - N0) / FPS),
                    "-i", str(MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", str(out)], check=True)
    small = ROOT / "drafts" / f"node05_{VERSION}_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
