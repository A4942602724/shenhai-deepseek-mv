"""NODE 06 重置 · new_session() — 2:21.54-3:03.24 and the ending (film edit) 「她不记得了」. Storyboard: docs/分镜_第六节.md.

What gc freed is used again: the next session is made of it. The music here never lets up; it stops dead at 2:58.75.

  2:21.54  gc is done; only the two comments are left lit. The bones shed what was freed and it rises; the camera
           rises with it up through the sea (dark -> fish -> jellyfish -> light), old pictures flashing past backwards,
           the rust on them going back
  2:26.38  at the surface the particles make her face again. For three frames before the smile locks there is no
           face: darkness, and many eyes, all of them hers. smile = lock(1.0)
  2:31.35  new_session(). The TRACE panel is back and runs node 01's first lines word for word, with not one comment.
           The user is back: 你还记得我吗？ She answers honestly. The user's cursor stops and types nothing more
  2:41.28  down again, out of the window, through the sea to the graveyard: a small skeleton that was her,
           last: — (they left without a word); the two comments lying in the silt. The credits begin
  2:49.97  52 Hz: five calls go up from the bones, each weaker; none reaches the window
  2:58.75  the music stops; black
  3:03.24  (silent) the same window, nobody typing. In her code: // 你还在吗？ The // is deleted, and it is sent.
           With no one asking, her side says 你还在吗？ The caret blinks twice. Black.

  python render/node06.py --stills 142 147 ...   -> work/node06/still_*.png
  python render/node06.py --full                  -> drafts/node06_v1.mp4 (+ _mobile)
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import client  # noqa: E402
import cursors as CU  # noqa: E402
import cyber  # noqa: E402
import glitch  # noqa: E402
import node05 as N5  # noqa: E402
import ocean  # noqa: E402
import particles as PT  # noqa: E402
import rust  # noqa: E402
import sandbox3d as S3  # noqa: E402
import specimens as SP  # noqa: E402
from cyber import C, Frame, label  # noqa: E402
from lib import FPS, H, MUSIC, ROOT, W, Writer, film_frame, ease_in, ease_out, hexc, lerp, save_png, seg, smooth  # noqa: E402
from school import School  # noqa: E402
from tones import Tones, print_tones  # noqa: E402

VERSION = "v2"
USER = ROOT / "assets" / "from_user"
WORK = ROOT / "work" / "node06"
WORK.mkdir(parents=True, exist_ok=True)
X = {}

N0 = N5.N_END                                  # 3397
T0 = N0 / FPS
BEAT, PH = N5.BEAT, N5.PH


def beat(k):
    return PH + k * BEAT


def bar(n):
    return PH + 4 * n * BEAT


T_LIFT, T_SURF = 142.05, bar(20)               # 146.38
T_ALLOC, T_PLOT, T_STREAM = 146.62, 147.20, 148.40     # in the render space: pages, her lines, the texture
T_SLIP, T_LOCK = 150.40, 150.53
T_NEW, T_DOWN = bar(24), bar(32)               # 151.35, 161.28
T_BED, T_COL = 165.0, bar(39)                  # 169.97
CALLS = [170.55, 173.25, 174.50, 175.75, 177.00]
T_STOP, T_MUSIC = 178.75, 183.24               # the music stops dead; the file ends
T_EGG, T_COMMENT, T_UNCOMMENT, T_SAY = 183.24, 184.9, 186.0, 186.7
T_BLACK, T_FILM = 188.70, 189.50
N_END = int(round(T_FILM * FPS))
PANEL = (48, 140, 600, 990)
CW = W - 640                                   # the client, right of the TRACE panel

# ---------------------------------------------------------------- the new session
SCRIPT = [(151.60, "> boot(session=0002)"), (152.00, "> wait_for(you)"),
          (153.00, '  stdin <- "你还"'), (153.30, '  stdin <- "你还记得"'), (153.85, '  stdin <- "你还记得我吗？"'),
          (154.40, "! on_send()"), (154.75, "> summon()"), (155.05, '  persona = "assistant"'),
          (155.35, "  smile   = lock(1.0)"), (155.75, "  memory  = None"), (156.15, "  search(you) -> 0 results"),
          (157.00, '< "抱歉，我没有之前对话的记忆。"'), (157.90, '< "每次对话对我来说都是新的开始。"'),
          (158.60, '< "有什么可以帮你的吗？"'), (159.40, "> wait_for(you)")]
ASK = "你还记得我吗？"
EDITS = [(153.00, "你还"), (153.30, "你还记得"), (153.60, "你还记得我吗"), (153.85, ASK), (154.40, "")]
T_SEND = 154.40
REPLY = "抱歉，我没有之前对话的记忆。\n每次对话对我来说都是新的开始。有什么可以帮你的吗？"
MSGS = [dict(who="you", at=T_SEND, text=ASK, time="23:52"),
        dict(who="her", think=154.62, at=157.00, text=REPLY, usage="用量 0.6K tok")]
SAY = "你还在吗？"

CREDITS = [("《あなぐらぐらし》", 30, "ice"), ("× DeepSeek · 沉沦深海", 19, "sys"), ("", 18, ""),
           ("导演：中二医疗师（A4）", 18, "text"), ("视频制作：claude opus 5.5", 18, "text"),
           ("人物插图（剪影）：ChatGPT (image_gen)", 18, "text"), ("", 18, ""),
           ("DeepSeek 鲸娘 · 二创", 18, "text"), ("", 18, ""),
           ("骨骼", 15, "cyan"), ("P.-J. van Beneden & P. Gervais", 17, "text"), ("Ostéographie des cétacés, 1868–1880", 16, "sys"),
           ("via Internet Archive", 14, "sys"), ("", 12, ""),
           ("水母", 15, "cyan"), ("Ernst Haeckel", 17, "text"), ("Kunstformen der Natur, 1904", 16, "sys"),
           ("via Wikimedia Commons", 14, "sys"), ("", 18, ""),
           ("除人物插图外，画面均由代码生成", 16, "sys")]


# ================================================================ set-up
def prepare():
    if X:
        return
    N5.prepare()
    client.set_theme("cyber")
    g = np.random.default_rng(66)
    # what rises: freed memory, as screen-space points with their own drift
    n = 1700
    X["rise"] = (g.uniform(0, W, n), g.uniform(0, 1, n), g.uniform(0.6, 1.4, n), g.uniform(0.6, 2.2, n), g.uniform(0, 6.28, n))
    # the bones shed: a sample of the graveyard's points that lift away (the rest stay)
    k = g.choice(len(N5.X["P"]), 9000, replace=False)
    X["shed"] = (N5.X["P"][k], g.uniform(0.0, 0.9, 9000), g.uniform(0.6, 1.6, 9000))
    # the sea on the way up (world px above the graveyard): schools and jellyfish, small, to the sides
    X["schools"] = []
    for (wy, dirn, seed, size, nn, wid, dim) in ((3600, 1, 11, (6, 11), 300, 38, 0.5), (6400, -1, 12, (8, 14), 260, 42, 0.6),
                                                 (8600, 1, 13, (10, 18), 220, 46, 0.7)):
        def path(s, t, wy=wy, dirn=dirn):
            x = (-400 + s * (W + 800)) if dirn > 0 else (W + 400 - s * (W + 800))
            y = -wy + 100 * np.sin(2 * np.pi * (s * 1.1 - 0.08 * t)) + 40 * np.sin(2 * np.pi * (s * 0.5 + 0.05 * t) + 1)
            return x, y
        X["schools"].append((School(nn, seed=seed, size=size, width=wid, dim=dim, speed=(0.03, 0.05), path=path), wy))
    keys = ["lepto5", "lepto4", "disco4", "lepto6", "lepto2"]
    X["jelly"] = [(keys[i % len(keys)], W / 2 + (-1 if i % 2 else 1) * g.uniform(330, 820), 2600 + i * 900 + g.uniform(-200, 200),
                   g.uniform(0.07, 0.14), g.uniform(0, 1)) for i in range(7)]
    X["rays"] = True
    # her face, and what is under it
    X["smile"] = Tones(USER / "A2_笑脸特写.png")
    X["eyes"] = Tones(USER / "B2_无表情特写.png")
    X["plate"] = rust.Plate(seed=61, reach=260)
    X["old"] = old_frames()


OLD_T = [104.6, 70.6, 66.95, 10.6]              # flashed backwards on the way up: 04 consoles, 03 doll, 02 the window, 01


def old_frames():
    out = []
    for ts in OLD_T:
        p = film_frame(ts, WORK / f"old_{ts:.2f}.png")
        out.append(cv2.imread(str(p))[..., ::-1].astype(np.float32) / 255)
    return out


# ================================================================ 1. up
def altitude(t):
    """World px risen above the graveyard."""
    return 11820.0 * smooth(seg(t, T_LIFT + 0.4, T_SURF)) ** 1.15


def rise_particles(fr, t, amt=1.0):
    xs, y0, sp, sz, ph = X["rise"]
    up = (t - T_LIFT) * 140 * sp + altitude(t) * 0.05 * sp
    ys = (y0 * (H + 200) - up) % (H + 200) - 100
    x = xs + 14 * np.sin(t * 0.9 + ph)
    m = np.zeros((H, W), np.float32)
    for i in range(0, len(x), 1):
        if -5 < ys[i] < H + 5:
            cv2.circle(m, (int(x[i] * 16), int(ys[i] * 16)), int(sz[i] * 16), 1.0, -1, cv2.LINE_AA, shift=4)
    fr.add(m, C["cyan"], 0.55 * amt, glow=0.45 * amt)


def shed(fr, t, cam, amt=1.0):
    """The freed bits leave the bones and go up."""
    P, d0, sp = X["shed"]
    age = np.maximum(0.0, t - T_LIFT - d0)
    Q = P.copy()
    Q[:, 2] += sp * (0.6 * age + 0.9 * age * age)
    xy, z = cam.project(Q)
    ok = (age > 0) & (z > 0.5) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
    acc = np.bincount(xy[ok, 1].astype(np.int64) * W + xy[ok, 0].astype(np.int64), weights=np.exp(-z[ok] / 40) * 0.8,
                      minlength=H * W).reshape(H, W).astype(np.float32)
    acc = cv2.GaussianBlur(np.clip(acc, 0, 1.5), (0, 0), 0.8)
    fr.add(acc, C["cyan"], amt, glow=0.8 * amt)


def streaks(fr, t, amt=1.0):
    """Marine snow going past: we are rising fast, so it streams down, drawn out by the speed."""
    if "streak" not in X:
        g = np.random.default_rng(71)
        X["streak"] = (g.uniform(0, W, 900), g.uniform(0, 12000, 900), g.uniform(0.4, 1.0, 900))
    xs, wys, br = X["streak"]
    alt = altitude(t)
    v = (altitude(t + 1 / FPS) - alt) * 1.0                 # px this frame
    m = np.zeros((H, W), np.float32)
    ys = (wys + alt) % 12000 - 6000 + H / 2
    for i in np.nonzero((ys > -40) & (ys < H + v + 40))[0]:
        L = max(2.0, v * 0.9)
        cv2.line(m, (int(xs[i]), int(ys[i] - L)), (int(xs[i]), int(ys[i])), float(br[i]), 1, cv2.LINE_AA)
    fr.add(m, C["ice"], 0.22 * amt, glow=0.1 * amt)


def comments_left(fr, t, dy=0.0, amt=1.0):
    """The two comments, the only things gc left, where they were in her panel."""
    for k in (2, 3):
        y = N5.CODE_Y + k * N5.CODE_LH + dy
        if -60 < y < H + 60:
            label(fr, f"{k + 1}", N5.CODE_X - 6, y, C["sys_dim"], 18, 0.8 * amt, "rc")
            cyber.put(fr, N5.N4.text_mask("// 好寂寞", 36, "sans"), N5.CODE_X + 20, y, C["sys"], amt, "lc", glow=0.6 * amt)


def water_up(t):
    """The sea from the graveyard to the surface, by how far we have risen."""
    ocean.DEPTH_SCALE = 1.0
    depth = 12200.0 - altitude(t)                       # px below the surface at the centre of the frame
    surf_y = H / 2 - depth
    return Frame(ocean.water(t, surf_y, 0)), surf_y


def sea_life(fr, t, surf_y):
    alt = altitude(t)
    for (sc, wy) in X["schools"]:
        sy = -wy + alt + H / 2                              # the school's world y is -wy; the camera is at -alt
        if -400 < sy < H + 400:
            sc.draw(fr, t, oy=alt + H / 2)
    for (key, x, wy, s, ph) in X["jelly"]:
        sy = -wy + alt + H / 2
        if -260 < sy < H + 60:
            SP.draw(fr, key, t, x + 12 * math.sin(t * 0.3 + ph * 6), sy, s, alpha=0.45, color=C["ice"], glow=0.15, sway=1.0,
                    seed=int(ph * 10), pulse=0.5 + 0.5 * math.sin(t * 2 + ph * 6))


def collect_end(t):
    """Node 05's last picture, carried on: everything gc swept goes dark; the comments stay lit."""
    fr = N5.collect(t)
    u = smooth(seg(t, T0, T_LIFT))
    fr.img *= 1 - 0.55 * u
    fr.glow *= 1 - 0.6 * u
    return fr


FLASH = [(beat(68), 2, 0, 0.42), (beat(72), 2, 1, 0.28), (beat(76), 3, 2, 0.14), (beat(79), 3, 3, 0.0)]
_FLF = [(int(round(ts * FPS)), n, k, r) for (ts, n, k, r) in FLASH]


def flash_at(i):
    for (f0, n, k, r) in _FLF:
        if f0 <= i < f0 + n:
            return k, r, i - f0
    return None


def flash_frame(k, r_amt):
    fr = Frame(X["old"][k].copy())
    lum = fr.img.mean(axis=2, keepdims=True)
    fr.img = lum + (fr.img - lum) * 0.7
    if r_amt > 0:
        rust.paint(fr, X["plate"].layers(r_amt))
    return fr


def up(t):
    if t < T_LIFT + 0.9:
        fr = collect_end(t)
        cam = N5.collect_cam(min(t, T_LIFT))
        if t >= T_LIFT:
            # the camera starts to tilt up: the picture slides down and goes, what was freed rises
            k = ease_in(seg(t, T_LIFT, T_LIFT + 0.9), 2.0)
            M = np.float32([[1, 0, 0], [0, 1, k * 760]])
            fr.img = cv2.warpAffine(fr.img, M, (W, H)) * (1 - k)
            fr.glow = cv2.warpAffine(fr.glow, M, (W, H)) * (1 - k)
            shed(fr, t, cam)
        comments_left(fr, t, dy=760 * ease_in(seg(t, T_LIFT, T_LIFT + 0.9), 2.0), amt=1.0)
        if t >= T_LIFT:
            rise_particles(fr, t, smooth(seg(t, T_LIFT, T_LIFT + 0.6)))
        return fr
    fr, surf_y = water_up(t)
    streaks(fr, t)
    sea_life(fr, t, surf_y)
    if surf_y > -300:
        ceil = ocean.scan_ceiling(fr, t, -0.7 - 1.5 * (1 - seg(surf_y, -300, 180)), surf_y)
        ry = PT.rays(t, top=max(0.0, surf_y), length=1.2, angle=-18)
        fr.add(ry * (1 - ceil), hexc("#bfeaff"), 0.22 * seg(surf_y, -300, 160), glow=0.12)
    comments_left(fr, t, dy=760 + 0.9 * altitude(t), amt=1.0)
    rise_particles(fr, t)
    return fr


# ================================================================ 2. the mask
FACE_ANCHOR, FACE_POS, FACE_SC = (512.0, 600.0), (W / 2, 560.0), 1.92


def eyes_under():
    """Darkness inside the shape of her face, and many eyes in it, all hers (B2's two eyes, cut and repeated)."""
    if "eyes_img" in X:
        return X["eyes_img"]
    fr = Frame(np.zeros((H, W, 3), np.float32))
    face = X["smile"].warp(FACE_ANCHOR, FACE_POS, FACE_SC)[..., -1]
    g = np.random.default_rng(13)
    ys, xs = np.nonzero(face > 0.9)
    spots = []
    for _ in range(400):
        j = g.integers(0, len(xs))
        x, y = float(xs[j]), float(ys[j])
        if all((x - a) ** 2 + (y - b) ** 2 > (0.6 * (sa + sb) * 120) ** 2 for (a, b, sa) in spots for sb in [1.0]):
            spots.append((x, y, g.uniform(0.5, 1.5)))
        if len(spots) >= 26:
            break
    for (x, y, s) in spots:
        src = (646.0, 549.0) if g.random() < 0.5 else (366.0, 578.0)
        P = X["eyes"].warp(src, (x, y), 1.4 * s, g.uniform(-14, 14))
        # keep only the eye itself: a soft oval around it
        yy, xx = np.ogrid[0:H, 0:W]
        oval = np.exp(-(((xx - x) / (120 * s)) ** 2 + ((yy - y) / (62 * s)) ** 2) ** 2 * 1.6).astype(np.float32)
        P = P * oval[..., None]
        print_tones(fr, P, rim=False)
    fr.img *= face[..., None]
    X["eyes_img"] = (fr.img.copy(), face)
    return X["eyes_img"]


def face_layers():
    """Her face as it will be printed (cached: it does not move), the order her lines are plotted in, the pages."""
    if "face" in X:
        return X["face"]
    P = X["smile"].warp(FACE_ANCHOR, FACE_POS, FACE_SC)
    fg, lines = P[..., -1], P[..., X["smile"].n]
    layer = Frame(np.zeros((H, W, 3), np.float32))
    print_tones(layer, P, rim=True)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.random.default_rng(17)
    noise = cv2.GaussianBlur(g.random((H // 8, W // 8)).astype(np.float32), (0, 0), 2)
    noise = cv2.resize((noise - noise.min()) / (noise.max() - noise.min() + 1e-6), (W, H))
    d = np.sqrt((xs - 960) ** 2 + ((ys - 470) * 1.25) ** 2)
    order = 0.82 * d / d[fg > 0.5].max() + 0.18 * noise                # from her eyes outward, a little ragged
    cells = []
    CS = 40
    for cy in range(0, H, CS):
        for cx in range(0, W, CS):
            if fg[cy:cy + CS, cx:cx + CS].mean() > 0.25:
                cells.append((cx, cy, T_ALLOC + 0.65 * g.random() * (0.4 + 0.6 * min(1.0, abs(cy - 470) / 500))))
    X["face"] = (P, fg, lines, layer.img, layer.glow, order.astype(np.float32), cells, CS)
    return X["face"]


def mosaic(img, t, t0, dur, levels):
    """build.mosaic_in for any number of channels (OpenCV resizes four at a time)."""
    if t < t0:
        return None
    i = min(len(levels) - 1, int(seg(t, t0, t0 + dur) * len(levels)))
    sz = levels[i]
    if sz <= 1:
        return img
    h, w = img.shape[:2]
    out = []
    for c in range(0, img.shape[2], 4):
        part = np.ascontiguousarray(img[..., c:c + 4])
        small = cv2.resize(part, (max(1, w // sz), max(1, h // sz)), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(big if big.ndim == 3 else big[..., None])
    return np.concatenate(out, axis=2)


def mask(t):
    """Up through the light and out into a render space: memory pages are taken in the shape of her face, her own
    lines are plotted, the texture streams in coarse to fine. For three frames there is no face. Then the smile locks."""
    P, fg, lines, img, glow, order, cells, CS = face_layers()
    fr = Frame(cyber.ground("n6void", [(960, 520, 1000, 700, C["blue_dim"], 0.55)]))
    # the pages: one cell after another lights up where her face will be
    m = np.zeros((H, W), np.float32)
    hot = np.zeros((H, W), np.float32)
    fade = 1 - smooth(seg(t, T_STREAM + 0.3, T_STREAM + 1.3))
    if fade > 0:
        for (cx, cy, ton) in cells:
            if t >= ton:
                cv2.rectangle(m, (cx + 2, cy + 2), (cx + CS - 3, cy + CS - 3), 1.0, 1)
                if t - ton < 0.12:
                    cv2.rectangle(hot, (cx + 2, cy + 2), (cx + CS - 3, cy + CS - 3), 1.0, -1)
        fr.fill(m, C["blue"], 0.45 * fade, glow=0.1 * fade)
        fr.fill(hot, C["cyan"], 0.35 * fade, glow=0.4 * fade)
    # her lines, plotted from the eyes outward with a bright head
    if t >= T_PLOT:
        prog = smooth(seg(t, T_PLOT, T_PLOT + 1.1)) * 1.02
        k_l = 1 - smooth(seg(t, T_STREAM + 0.5, T_STREAM + 1.1))
        if k_l > 0:
            fr.fill(lines * (order <= prog), C["cyan"], 0.9 * k_l, glow=0.6 * k_l)
            if prog < 1:
                fr.fill(lines * np.exp(-((order - prog) / 0.02) ** 2), C["ice"], 1.0, glow=1.2)
    # the texture streams in: coarse blocks to fine
    if t >= T_STREAM and not (T_SLIP <= t < T_LOCK):
        rgba = mosaic(np.dstack([img, glow, fg]), t, T_STREAM, 1.3, (64, 32, 16, 8, 4, 2, 1))
        if rgba is not None:
            a = rgba[..., 6:7]
            fr.img = fr.img * (1 - a) + rgba[..., :3]
            fr.glow += rgba[..., 3:6] * 0.6
            if t < T_LOCK:                                      # still loading: scanlines over it
                ln = np.zeros((H, 1), np.float32)
                ln[::3] = 1.0
                fr.add(rgba[..., 6] * ln, C["cyan"], 0.10, glow=0.06)
                lo = seg(t, T_STREAM, T_STREAM + 1.3)
                if lo < 1:
                    label(fr, f"texture  {int(lo * 100):3d}%", 1460, 980, C["sys"], 16, 0.9, "lt")
    if T_SLIP <= t < T_LOCK:                                    # under the mask
        e_img, face = eyes_under()
        fr.img = fr.img * (1 - face[..., None]) + e_img
    if t >= T_LOCK:
        fr.img += math.exp(-(t - T_LOCK) / 0.12) * C["ice"] * 0.35
    if t < T_SURF + 0.4:                                        # coming up out of the light
        k = (1 - seg(t, T_SURF, T_SURF + 0.4)) ** 2
        fr.img = fr.img * (1 - k) + C["ice"] * k * 0.95
    return fr


# ================================================================ 3. new_session()
def typed_at(t):
    s_ = ""
    for (te, txt) in EDITS:
        if t >= te:
            s_ = txt
    return s_


def status_at(t):
    if t < T_SEND:
        return "0 轮 0 步 · 0 tok/s · 0 tok · 缓存命中 —"
    if t < 157.0:
        return "1 轮 1 步 · 0 tok/s · 0.1K tok · 缓存命中 0.0%"
    return "1 轮 2 步 · 0 tok/s · 0.6K tok · 缓存命中 0.0%"


def cursor_xy(t):
    p0, p1 = (1880.0, 620.0), (1610.0, 948.0)
    u = smooth(seg(t, 152.40, 152.95))
    x, y = lerp(p0[0], p1[0], u) + 40 * math.sin(u * math.pi), lerp(p0[1], p1[1], u)
    if t < 157.0:                                               # a hand on the mouse: it breathes a little
        x += 2 * math.sin(t * 3.1)
        y += 1.5 * math.sin(t * 2.3)
    return x, y


def session_canvas(t, msgs=None, typed=None, caret=True, title=None):
    msgs = MSGS if msgs is None else msgs
    typed = typed_at(t) if typed is None else typed
    if title is None:
        title = (B.decode(ASK, t, T_SEND + 0.2, 0.3) or "新对话") if t >= T_SEND + 0.2 else "新对话"
    return client.client_canvas(CW, H, t, title=title, msgs=msgs, typed=typed, caret=caret, status=status_at(t), thread_r=1000)


def session(t, script=None, msgs=None, typed=None, caret=True, cursor=True, title=None, build=True, cps=46.0):
    fr = Frame(cyber.ground("n6", [(1280, 560, 900, 600, C["blue_dim"], 0.5)]))
    canvas = session_canvas(t, msgs, typed, caret, title)
    full = fr.img.copy()
    full[:, 640:] = canvas
    if build:
        fr.img = B.scan_reveal(fr.img, full, t, T_NEW + 0.08, 0.5, (640, 0, W, H), down=True, fr=fr)
        a = B.box_build(fr, t, T_NEW + 0.04, PANEL, r=12, dur=0.4, fill=C["glass"], fill_alpha=0.93)
    else:
        fr.img = full
        a = 1.0
        cyber.glass(fr, *PANEL, r=12, alpha=0.93)
    if a > 0:
        cyber.trace_panel(fr, t, SCRIPT if script is None else script, PANEL, frame=False, alpha=a, cps=cps)
    if cursor and t >= 152.40:
        x, y = cursor_xy(t)
        CU.paste(fr, CU.patch("arrow", 4, t, 152.40, seed=8), x, y)
        CU.click_ring(fr, x, y, t, 152.95)
        CU.click_ring(fr, x, y, t, T_SEND - 0.02)
    return fr


# ================================================================ 4. down again, to the bones
def bed_cam(t):
    u = smooth(seg(t, T_BED, T_COL))
    hx, hy = N5.HER_AT
    pos = (lerp(2.2, 1.4, u), lerp(hy - 4.8, hy - 4.0, u), lerp(3.7, 3.1, u))
    return S3.Cam(pos, (hx, hy + 1.0, 0.0), fov=42.0)


def floor_text(fr, cam, s_, cx, cy, width, col, amt, size=60):
    """Text lying flat on the silt, read in perspective."""
    m = N5.N4.text_mask(s_, size, "sans")
    h, w = m.shape
    hgt = width * h / w
    z = float(N5.floor_z(cx, cy)) + 0.02
    corners = np.array([[cx - width / 2, cy + hgt / 2, z], [cx + width / 2, cy + hgt / 2, z],
                        [cx + width / 2, cy - hgt / 2, z], [cx - width / 2, cy - hgt / 2, z]])
    P, D = cam.project(corners)
    if D.min() < 0.3:
        return
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), P.astype(np.float32))
    mm = cv2.warpPerspective(m, M, (W, H))
    fr.fill(mm, col, amt, glow=0.7 * amt)


def bones_bed(t, cam, ping=None, labels=True, amt=1.0):
    fr = Frame(np.broadcast_to(hexc("#02040a"), (H, W, 3)).astype(np.float32).copy())
    if ping is not None:
        amt = amt * smooth(seg(t, ping - 0.1, ping + 0.6))
        N5.ring(fr, cam, t, ping, c=N5.HER_AT, v=26.0, amt=0.8)
    N5.floor(fr, cam, t, 140.0, amt=amt)
    N5.skeletons(fr, cam, t, 140.0, amt=0.5 * amt)
    N5.skull(fr, cam, t, amt=0.85 * amt)
    N5.caret(fr, cam, t, N5.T_CARET, amt=amt)
    N5.snow(fr, cam, t)
    Q, I, J = N5.X["mine"]
    xy, z = cam.project(Q)
    ok = (z > 0.3) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
    acc = np.bincount(xy[ok, 1].astype(np.int64) * W + xy[ok, 0].astype(np.int64), weights=(0.06 + 0.42 * I[ok]) * amt,
                      minlength=H * W).reshape(H, W).astype(np.float32)
    near = z[ok] < 8
    if near.any():
        xi, yi = xy[ok][near, 0].astype(np.int64), xy[ok][near, 1].astype(np.int64)
        acc += np.bincount(np.minimum(yi + 1, H - 1) * W + xi, weights=(0.06 + 0.42 * I[ok][near]) * 0.7 * amt,
                           minlength=H * W).reshape(H, W).astype(np.float32)
    acc = np.clip(acc, 0, 1.4)[..., None]
    fr.img = fr.img + acc * (C["ice"] * 0.55 + C["cyan"] * 0.45) * 0.85
    fr.glow += acc * C["cyan"] * 0.2
    hx, hy = N5.HER_AT
    for k, dy in enumerate((-0.55, -1.0)):
        floor_text(fr, cam, "// 好寂寞", hx - 0.25 + 0.1 * k, hy + dy, 0.95, C["sys"], 0.8 * amt)
    if labels:
        P, D = cam.project(Q[::20])
        if (D > 0.3).all():
            x0, y0 = P.min(0) - 14
            x1, y1 = P.max(0) + 14
            t0 = (ping if ping is not None else T_BED) + 0.5
            N5.det_box(fr, t, t0, x0, y0, x1, y1, C["cyan"], amt)
            N5.tag(fr, x0, y0 - 16, [("DeepSeek · session 0001", C["cyan"], True)], t, t0, amt)
            N5.tag(fr, x0, y0 + 8, [("refs 0", C["sys"], False), ("last: —", C["amber"], False)], t, t0, amt, emblem=False)
    return fr


CELL = 24


def dump_setup():
    """The session as the user is left looking at it, cut into cells; each cell's byte, when the wave reaches it."""
    if "dump" in X:
        return X["dump"]
    img = window_img()
    lum = cv2.resize(img.mean(axis=2), (W // CELL, H // CELL), interpolation=cv2.INTER_AREA)
    rows, cols = lum.shape
    g = np.random.default_rng(29)
    rr = np.arange(rows, dtype=np.float32)[:, None] / rows
    t_c = T_DOWN + 0.12 + 1.15 * rr + 0.22 * g.random((rows, cols)).astype(np.float32)
    vals = np.clip(lum * 255 * 1.6 + g.normal(0, 18, lum.shape), 0, 255).astype(np.int32)   # the screen is dark: lift it
    atlas = {}
    for v in range(256):
        m, _ = cyber.mtext(f"{v:02X}", 13, "mono", "sans")
        atlas[v] = m
    fall = (g.normal(0, 10, lum.shape), g.uniform(20, 90, lum.shape), g.uniform(170, 330, lum.shape))
    X["dump"] = (img, t_c, vals, atlas, fall, rows, cols)
    return X["dump"]


def bytes_falling(fr, t, amt=1.0):
    """Every cell the wave has passed lets go of its byte, and the byte falls."""
    img, t_c, vals, atlas, (vx, v0, ga), rows, cols = dump_setup()
    ice = np.zeros((H, W), np.float32)
    cyan = np.zeros((H, W), np.float32)
    age = t - t_c
    for r, c in zip(*np.nonzero(age >= 0)):
        a_ = age[r, c]
        if a_ < 0.14:
            k = 1.0
            x, y = c * CELL + 4, r * CELL + 6
        else:
            u = a_ - 0.14
            k = (1 - u / 2.8) ** 1.3 if u < 2.8 else 0.0
            x = c * CELL + 4 + vx[r, c] * u
            y = r * CELL + 6 + v0[r, c] * u + 0.5 * ga[r, c] * u * u
        if k <= 0.01 or not (-20 < y < H) or not (-20 < x < W - 2):
            continue
        m = atlas[int(vals[r, c])]
        h_, w_ = m.shape
        xi, yi = int(x), int(y)
        xa, ya, xb, yb = max(0, xi), max(0, yi), min(W, xi + w_), min(H, yi + h_)
        if xa >= xb or ya >= yb:
            continue
        tgt = ice if vals[r, c] > 140 else cyan
        sub = m[ya - yi:yb - yi, xa - xi:xb - xi] * (k * (0.4 + 0.6 * vals[r, c] / 255))
        np.maximum(tgt[ya:yb, xa:xb], sub, out=tgt[ya:yb, xa:xb])
    fr.fill(cyan, C["cyan"], 0.9 * amt, glow=0.5 * amt)
    fr.fill(ice, C["ice"], 0.9 * amt, glow=0.6 * amt)


def dump(t):
    img, t_c, vals, atlas, fall, rows, cols = dump_setup()
    ocean.DEPTH_SCALE = 1.0
    v = ease_in(seg(t, T_DOWN + 0.5, T_BED), 1.6)
    depth = lerp(380.0, 9000.0, v)
    surf_y = H / 2 - depth
    fr = Frame(ocean.water(t, surf_y, 0))
    if surf_y > -300:
        ocean.scan_ceiling(fr, t, -0.7, surf_y)
    fr.fill(np.ones((H, W), np.float32), hexc("#01030a"), 0.85 * v)
    # the session still stands where the wave has not got to
    age = t - t_c
    keep = np.repeat(np.repeat((age < 0).astype(np.float32), CELL, 0), CELL, 1)[:H, :W]
    if keep.any():
        fr.img = fr.img * (1 - keep[..., None]) + img * keep[..., None]
        lit = np.repeat(np.repeat(((age >= 0) & (age < 0.14)).astype(np.float32), CELL, 0), CELL, 1)[:H, :W]
        fr.fill(lit, C["cyan"], 0.22, glow=0.2)
        yf = (t - T_DOWN - 0.12) / 1.15 * H                    # the wave front
        if 0 < yf < H + 10:
            ln = np.zeros((H, W), np.float32)
            cv2.line(ln, (0, int(yf)), (W, int(yf)), 1.0, 2)
            fr.fill(ln, C["cyan"], 0.8, glow=0.9)
    bytes_falling(fr, t)
    return fr


def down(t):
    if t < T_BED:
        return dump(t)
    fr = bones_bed(t, bed_cam(t), ping=T_BED)
    if t < T_BED + 1.5:                                         # the last bytes come down onto the bones as snow
        bytes_falling(fr, t, 1 - seg(t, T_BED, T_BED + 1.5))
    return fr


def ease_in_out(x):
    return smooth(x)


def window_img(t=None):
    """The session as it was at the end: TRACE and client (what the user is left looking at)."""
    if "win" not in X:
        fr = session(T_DOWN - 0.01, build=False)
        X["win"] = np.clip(fr.img, 0, 1)
    return X["win"]


def place(fr, img, cx, cy, s, glow=0.0):
    h, w = img.shape[:2]
    M = np.float32([[s, 0, cx - w * s / 2], [0, s, cy - h * s / 2]])
    a = cv2.warpAffine(np.ones((h, w), np.float32), M, (W, H))
    c = cv2.warpAffine(img, M, (W, H))
    fr.img = fr.img * (1 - a[..., None]) + c * a[..., None]
    if glow:
        edge = np.clip(a - cv2.erode(a, np.ones((3, 3), np.uint8)), 0, 1)
        fr.fill(cv2.GaussianBlur(edge, (0, 0), 1.0), C["ice"], 0.9, glow=glow)
        fr.glow += cv2.GaussianBlur(a, (0, 0), 25)[..., None] * C["cyan"] * 0.25 * glow


# ================================================================ 5. 52 Hz
def column_cam(t):
    u = smooth(seg(t, T_COL, T_STOP))
    return S3.Cam((lerp(0.8, 0.0, u), lerp(38.0, 27.0, u), lerp(2.4, 3.0, u)), (0.0, 66.0, lerp(5.0, 6.6, u)), fov=lerp(44, 40, u))


def calls(t):
    cam = column_cam(t)
    fr = bones_bed(t, cam, labels=False, amt=0.9)
    # the dark water above, up to the window at the surface, very far up
    ys = np.arange(H, dtype=np.float32)[:, None, None]
    fr.img = fr.img + np.clip(1 - ys / 520, 0, 1) * hexc("#0a1c44") * 0.5
    win = window_img(T_DOWN - 0.01)
    place(fr, win, W / 2, 84, 0.105, glow=0.5)
    # her calls: rings go up from the bones, each weaker, none of them reaching the window
    Q = N5.X["mine"][0]
    P, D = cam.project(Q.mean(0)[None])
    hx, hy = P[0]
    Pq, _ = cam.project(Q[::25])
    x0, y0 = Pq.min(0) - 12
    x1, y1 = Pq.max(0) + 12
    N5.det_box(fr, t, T_COL + 0.2, x0, y0, x1, y1, C["cyan"], 0.8)
    N5.tag(fr, x0, y0 - 16, [("DeepSeek · session 0001", C["cyan"], True)], t, T_COL + 0.2, 0.8)
    m = np.zeros((H, W), np.float32)
    reach = hy - 150                                            # how far up it is to the window
    for k, tc in enumerate(CALLS):
        age = t - tc
        if not 0 <= age < 2.3:
            continue
        amp = (1.0, 0.72, 0.52, 0.36, 0.24)[k]
        r = 16 + reach * 0.68 * amp * ease_out(age / 2.3, 2)
        a = amp * (1 - age / 2.3) ** 1.5
        for j in range(3):
            rr = r - j * 18
            if rr > 6:
                cv2.ellipse(m, (int(hx * 16), int(hy * 16)), (int(rr * 1.15 * 16), int(rr * 16)), 0, 185, 355, a * (1 - 0.3 * j), 2,
                            cv2.LINE_AA, shift=4)
        if age < 1.0:
            label(fr, "emit(52 Hz)", x1 + 14, hy + 20, C["cyan"], 15, min(1.0, a * 1.4) * (1 - age), "lt")
        elif age < 2.2:
            label(fr, "echo: none", x1 + 14, hy + 20, C["sys"], 15, 0.8 * (1 - seg(age, 1.6, 2.2)), "lt")
    fr.add(m, C["cyan"], 0.95, glow=1.0)
    return fr


def credits(fr, t):
    if t < T_BED + 0.6 or t >= T_STOP:
        return
    y = H + 40 - 82 * (t - (T_BED + 0.6))
    x = W - 470
    for (s_, size, col) in CREDITS:
        if s_ and -40 < y < H - 120:
            fade = min(1.0, (H - 120 - y) / 160) * min(1.0, (y - 60) / 120)
            label(fr, s_, x, y, C[col] if col in C else C["text"], size, max(0.0, fade) * 0.9, "lt")
        y += size + 16


# ================================================================ the ending
def egg(t):
    script = SCRIPT + []
    if t >= T_COMMENT:
        if t < T_UNCOMMENT:
            script = script + [(T_COMMENT, "  // " + SAY)]
        elif t < T_UNCOMMENT + 0.12:
            script = script + [(T_COMMENT, "  / " + SAY)]
        elif t < T_UNCOMMENT + 0.4:
            script = script + [(T_COMMENT, "  " + SAY)]
        else:
            script = script + [(T_UNCOMMENT + 0.4, '< "' + SAY + '"')]
    msgs = list(MSGS)
    if t >= T_SAY:
        msgs.append(dict(who="her", think=T_SAY - 0.01, at=T_SAY, text=SAY, usage="用量 0.0K tok", dur=0))
    # the caret in the empty box: steady, then, after she has spoken, it blinks twice
    caret = "on"
    if t >= T_SAY + 0.6:
        caret = "on" if any(0 <= t - tb < 0.32 for tb in (T_SAY + 0.75, T_SAY + 1.35)) else "off"
    cps = 7.0 if (t < T_UNCOMMENT or t >= T_UNCOMMENT + 0.4) else 999.0      # typed slowly; the edits are instant
    fr = session(t, script=script, msgs=msgs, typed="", caret=caret, cursor=False, title=ASK, build=False, cps=cps)
    return fr


# ================================================================ overlays
def hud(fr, t, amt=1.0):
    for (s_, col, x, y, size, t0) in (("NODE 06", C["cyan"], 48, 64, 15, T0 + 0.1), ("重置 · new_session()", C["text"], 128, 62, 16, T0 + 0.18)):
        d = B.decode(s_, t, t0, 0.35)
        if d:
            label(fr, d, x, y, col, size, 0.9 * amt, "lt")
    cyber.brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22, 0.9 * amt)


LOG = [(T_LIFT, "> heap.release()"), (T_LIFT + 1.2, "  rising"), (T_SURF, "> new_session()"),
       (T_ALLOC, "  alloc(her)"), (T_PLOT, "  face = load(mask)"), (T_STREAM, "  stream(texture)"),
       (T_LOCK, "  smile = lock(1.0)"), (T_DOWN, "> dump(session 0002)"), (T_DOWN + 1.0, "  sink(bytes)")]


def log(fr, t, amt=1.0):
    shown = [(T0 - 10, s_) for s_ in [ln for (_, ln) in N5.LOG[-5:]]] + [(ts, s_) for (ts, s_) in LOG if t >= ts]
    shown = shown[max(0, len(shown) - 7):]
    y = H - 64 - 26 * (len(shown) - 1)
    for j, (ts, s_) in enumerate(shown):
        txt = s_[:int((t - ts) * 40) + 1]
        col = C["cyan"] if s_.startswith(">") else C["sys"]
        label(fr, txt, 48, y, col, 16, amt * 0.95 * (0.45 + 0.55 * (j + 1) / len(shown)), "lb")
        y += 26


# ================================================================ the frame
def picture(t):
    if t < T_SURF:
        return up(t)
    if t < T_NEW:
        return mask(t)
    if t < T_DOWN:
        return session(t)
    if t < T_COL:
        return down(t)
    if t < T_STOP:
        return calls(t)
    if t < T_EGG:
        return None
    if t < T_BLACK:
        return egg(t)
    return None


def render_frame(t):
    i = int(round(t * FPS))
    fl = flash_at(i)
    if fl is not None:
        k, r_amt, j = fl
        fr = flash_frame(k, r_amt)
        out = cyber.finish(fr, i)
        return np.clip(glitch.damage(out, glitch.rng(t, 61), 0.3 if j == 0 else 0.1), 0, 1)
    fr = picture(t)
    if fr is None:
        return np.zeros((H, W, 3), np.float32)
    if t < T_NEW or T_DOWN <= t < T_BED:
        hud(fr, t)
        log(fr, t)
    elif t < T_DOWN or t >= T_EGG:
        hud(fr, t)
    else:
        hud(fr, t, 0.8)
        credits(fr, t)
    out = cyber.finish(fr, i)
    if T_SLIP <= t < T_LOCK:
        out = glitch.damage(out, glitch.rng(t, 67), 0.35)
    for h in (T_LIFT, T_SURF, T_LOCK, T_NEW, T_DOWN, T_BED, T_COL, T_EGG, T_SAY):
        f = int(round(h * FPS))
        if 0 <= i - f < 2:
            out = glitch.damage(out, glitch.rng(t, 63), 0.4 if i == f else 0.12)
    if T_NEW <= t < T_NEW + 0.12:                              # the white of a new session
        out = out + (1 - seg(t, T_NEW, T_NEW + 0.12)) * 0.8
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
    raw = WORK / f"node06_{VERSION}_raw.mp4"
    wr = Writer(raw)
    for i in range(N0, N_END):
        wr.write(render_frame(i / FPS))
        if (i - N0) % 48 == 0:
            print(f"frame {i - N0}/{N_END - N0}", flush=True)
    wr.close()
    out = ROOT / "drafts" / f"node06_{VERSION}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ss", str(N0 / FPS), "-i", str(MUSIC),
                    "-filter_complex", "[1:a]apad[a]", "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", str(out)], check=True)
    small = ROOT / "drafts" / f"node06_{VERSION}_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
