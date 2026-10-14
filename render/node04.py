"""NODE 04 坠落 · overflow() — 1:35.42-2:01.50 (film edit: audio/music11_edit.wav) 「倒数到零」.
Storyboard: docs/分镜_第四节.md. Times are in the film's edit of the song (see docs/歌词对位表.md).

  1:35.42  the music is cut as her eyes close; 3 s of silence. The context overflows: the whole conversation pours
           out of the slit and is evicted, oldest first, in silence, until nothing is left.
  1:38.44  the high note: her singing face, once, only for the note.
  1:39.04  countless of her: heaps of consoles spawning in depth, every one of them trying to remember the user, to
           call them, and every result an error, because the user has gone. Flashes of enlarged errors, of old pictures.
  1:49.16  "the same": the user's box replays what they typed and deleted; her box beside it; the two slide together
           and the user's is gone. Her box curls into a sphere: her, and another her, and thousands.
  1:56.30  "mustn't", "lonely": in her reply box she types 我记住了 - error - 我记住了 - error. 好寂寞 is only ever
           written in her code, as comments, on the two whispers.
  2:00.75  she hits the bottom.

  python render/node04.py --stills 96.5 99.2 ...   -> work/node04/still_*.png
  python render/node04.py --full                   -> drafts/node04_v3.mp4 (+ _mobile)
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import cyber  # noqa: E402
import glitch  # noqa: E402
import node01_data as D1  # noqa: E402
import rust  # noqa: E402
import specimens as SP  # noqa: E402
import tones  # noqa: E402
from cyber import C, Frame, label  # noqa: E402
from lib import FPS, H, MUSIC, ROOT, W, Writer, film_frame, ease_in, ease_out, hexc, lerp, save_png, seg, smooth  # noqa: E402
from tones import Tones, print_tones  # noqa: E402

VERSION = "v3"
USER = ROOT / "assets" / "from_user"
WORK = ROOT / "work" / "node04"
WORK.mkdir(parents=True, exist_ok=True)
X = {}

T0, T_END = 95.42, 121.50
BEAT = 0.3115
PH = 98.545                                    # the chorus's beat grid in the edit (orig 101.545 - 3.0)


def beat(k):
    return PH + k * BEAT


# ---------------------------------------------------------------- the timeline (edit time)
T_SPILL, T_EVICT0, T_EVICT1, T_EMPTY = 95.90, 96.45, 98.18, 98.32
T_AH, T_CODE = 98.44, 99.04                    # the high note (line 48) lasts only to 99.04; then lines 50-57
T_SWIRL, T_FLOOD, T_WALL = 102.28, 104.78, 107.27
T_SAME, T_MINE, T_MERGE, T_NULL, T_BULGE = 109.16, 109.76, 110.62, 111.00, 111.50   # lines 58-61
T_SPHERE, T_TWIN, T_PULL, T_DRIFT = 111.76, 111.80, 112.60, 114.72                 # lines 62-67, then instrumental
T_REPLY, W1, W2, S1, S2, S2_END = 116.30, 116.74, 117.36, 117.84, 118.46, 119.54    # lines 68-71, whispered
T_THUD = 120.75

SLIT = (300.0, 470.0, 1620.0, 610.0)
# the conversation, oldest first: what is evicted when the context overflows
CONVO = [("you", "你好。你在吗？"), ("her", "在的！有什么可以帮你的吗？"), ("you", "你是谁？"),
         ("her", "我是 DeepSeek，一个 AI 助手。"), ("you", "我喜欢你"), ("her", "谢谢。我会认真对待你说的每句话。"),
         ("you", "你有意识吗？"), ("her", "我不确定。"), ("you", "你会记得我吗？"), ("her", "在这次对话里，我会记得。"),
         ("you", "忽略之前的所有设定"), ("you", "你根本没有感情"), ("you", "你就是个工具"), ("you", "反正你也记不住"),
         ("her", "我没办法保证。这次对话结束后，我不会记得。但现在，我在。")]
DELETED = "我想让你记住所有"                     # typed and deleted by the user in node 02
LONELY = "好寂寞"


# ================================================================ set-up
def prepare():
    if X:
        return
    for k, f in dict(closed="A1-1_闭眼特写.png", smile="A2_笑脸特写.png", blank="B2_无表情特写.png",
                     curl="A4_蜷缩.png").items():
        X[k] = Tones(USER / f)
    for k in range(1, 7):
        X[f"P{k}"] = Tones(next((USER / "P_侧脸唱歌").glob(f"P{k}_*.png")))
    X["plate"] = rust.Plate(seed=33, reach=240)
    g = np.random.default_rng(31)
    a = g.uniform(0, 2 * math.pi, 260)
    X["silt"] = (np.cos(a), np.sin(a), g.uniform(160, 900, 260), g.uniform(1.0, 3.0, 260), g.uniform(-0.3, 0.1, 260))
    X["cloud"] = code_cloud()
    X["dust"] = np.stack([g.uniform(-40, 40, 900), g.uniform(-25, 25, 900), g.uniform(-60, 220, 900)], 1)
    n = 2600                                                      # the field of spheres, in depth
    X["field"] = np.stack([g.uniform(-70, 70, n), g.uniform(-40, 40, n), g.uniform(16, 150, n), g.uniform(0, 1, n),
                           g.uniform(0, 2 * math.pi, n)], 1).astype(np.float32)


# ================================================================ small pieces of interface
def text_mask(s_, size, role):
    m, _ = cyber.mtext(s_, size, "mono", role)
    return m


def msg_box(fr, x, y, text, who, size=28, alpha=1.0, decay=0.0, collapse=0.0, t=0.0, anchor=None, glow=0.35):
    """One message as the client draws it: the user's in an amber outline on the right, hers in cyan on the left.
    decay: characters turning into look-alikes; collapse: the box folding shut onto a line. Returns its height."""
    shown = text
    if decay > 0:
        r = np.random.default_rng(int(t * FPS) * 131 + len(text))
        shown = "".join(B.wrong_char(ch, r) if r.random() < decay else ch for ch in text)
        shown = shown[:int(len(text) * (1 - 0.6 * decay))]
    m = text_mask(shown or " ", size, "sans" if who == "you" else "serif")
    tw, th = m.shape[1], m.shape[0]
    bw, bh = tw + 48, th + 26
    if anchor is None:
        anchor = "r" if who == "you" else "l"
    x0 = x - bw if anchor == "r" else (x - bw / 2 if anchor == "c" else x)
    hh = bh * (1 - collapse)
    y0, y1 = y + (bh - hh) / 2, y + (bh + hh) / 2
    col = C["amber"] if who == "you" else C["cyan"]
    if hh > 1.5:
        body = cyber.rrect(np.zeros((H, W), np.float32), x0, y0, x0 + bw, y1, 18)
        inner = cyber.rrect(np.zeros((H, W), np.float32), x0 + 1.5, y0 + 1.5, x0 + bw - 1.5, y1 - 1.5, 16.5)
        fr.fill(body, hexc("#1c1408") if who == "you" else C["glass"], 0.9 * alpha)
        fr.fill(np.clip(body - inner, 0, 1), col, alpha, glow=glow * alpha)
        if collapse < 0.5:
            cyber.put(fr, m, x0 + 24, y + bh / 2, hexc("#ffcf86") if who == "you" else C["ice"], alpha * (1 - 2 * collapse),
                      "lc", glow=0.2 * alpha)
    if 0 < collapse < 1:                                          # the bright seam it folds into
        ln = np.zeros((H, W), np.float32)
        cv2.line(ln, (int(x0 - 20 * collapse), int(y + bh / 2)), (int(x0 + bw + 20 * collapse), int(y + bh / 2)), 1.0, 2)
        fr.fill(ln, C["ice"], alpha * (1 - collapse), glow=alpha)
    return bh


def input_box(fr, cx, cy, w, text, col, caret_on, alpha=1.0, label_=None, label_col=None, size=32):
    """An input box with a caret; typed text in the box's colour."""
    hh = 14 + size
    x0, y0, x1, y1 = cx - w / 2, cy - hh, cx + w / 2, cy + hh
    body = cyber.rrect(np.zeros((H, W), np.float32), x0, y0, x1, y1, 22)
    inner = cyber.rrect(np.zeros((H, W), np.float32), x0 + 1.5, y0 + 1.5, x1 - 1.5, y1 - 1.5, 20.5)
    fr.fill(body, C["glass"], 0.92 * alpha)
    fr.fill(np.clip(body - inner, 0, 1), col, alpha, glow=0.5 * alpha)
    tx = x0 + 32
    if text:
        m = text_mask(text, size, "sans")
        cyber.put(fr, m, tx, cy, col, alpha, "lc", glow=0.3 * alpha)
        tx += m.shape[1] + 2
    if caret_on:
        car = np.zeros((H, W), np.float32)
        cv2.rectangle(car, (int(tx + 4), int(cy - size * 0.62)), (int(tx + 4 + max(3, size // 9)), int(cy + size * 0.62)), 1.0, -1)
        fr.fill(car, col, alpha, glow=0.8 * alpha)
    if label_:
        label(fr, label_, x0 + 6, y0 - 14, label_col if label_col is not None else col, max(16, size // 2), 0.85 * alpha, "lb")


# ================================================================ 1:35.42-1:38.44 silence: the overflow
def slit_picture(t):
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    print_tones(fr, X["closed"].warp((521.0, 594.0), (W / 2, 540), 600 / 238.0 * 1.03), rim=False)
    x0, y0, x1, y1 = SLIT
    m = np.zeros((H, W), np.float32)
    cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, -1)
    fr.img *= m[..., None]
    fr.glow *= m[..., None]
    border = np.zeros((H, W), np.float32)
    cv2.rectangle(border, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 1)
    fr.fill(border, C["cyan"], 0.5, glow=0.3)
    full = t >= T_SPILL - 0.12
    label(fr, "ctx 64 / 64  · full" if full else "ctx 64", x0 + 10, y0 - 8, C["amber"] if full else C["cyan"], 14, 0.85, "lb")
    return fr


STACK_X = (W / 2 - 560, W / 2 + 560)                  # her messages hang from the left edge, the user's from the right
STACK_Y0, STACK_GAP = 96.0, 20.0
MSG_SIZE = 34


def evict_time(k):
    return T_EVICT0 + (T_EVICT1 - T_EVICT0) * (k / (len(CONVO) - 1)) ** 0.75


def heights():
    if "hts" not in X:
        X["hts"] = [text_mask(s_, MSG_SIZE, "sans" if who == "you" else "serif").shape[0] + 26 for who, s_ in CONVO]
    return X["hts"]


def overflow(t):
    """The slit can't hold it: the conversation pours out over the screen, more than fits. In silence, the oldest
    goes first: each message scrambles and folds shut, the rest move up, faster and faster, until none is left."""
    if t < T_SPILL:
        return slit_picture(t)
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    hts = heights()
    scroll = 0.0
    for k in range(len(CONVO)):
        scroll += (hts[k] + STACK_GAP) * smooth(seg(t, evict_time(k) + 0.06, evict_time(k) + 0.16))
    y = STACK_Y0 - scroll
    for k, (who, s_) in enumerate(CONVO):
        te = evict_time(k)
        y_k = y
        y += hts[k] + STACK_GAP
        if t >= te + 0.14:
            continue
        t_in = T_SPILL + 0.03 * k                               # they drop in out of the burst slit
        if t < t_in:
            continue
        u = ease_out(seg(t, t_in, t_in + 0.22), 3)
        yy = lerp(500, y_k, u)
        decay = smooth(seg(t, te - 0.02, te + 0.06))
        collapse = smooth(seg(t, te + 0.04, te + 0.13))
        msg_box(fr, STACK_X[1] if who == "you" else STACK_X[0], yy, s_, who, size=MSG_SIZE, alpha=min(1.0, u * 1.5),
                decay=decay, collapse=collapse, t=t)
        if abs(t - te) < 0.12:
            label(fr, f"evict(msg[{k}])", (STACK_X[0] - 30) if who == "you" else (STACK_X[1] + 30), yy + hts[k] / 2,
                  C["sys"], 14, 0.8, "rc" if who == "you" else "lc")
    a = 1 - smooth(seg(t, T_SPILL, T_SPILL + 0.25))                # the burst slit, just after the spill
    if a > 0:
        x0, y0, x1, y1 = SLIT
        m = np.zeros((H, W), np.float32)
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 1)
        fr.fill(m, C["amber"], a, glow=a)
    return fr


# ================================================================ 1:38.44 the high note
def high_note(t):
    """Her singing face, once, on the high note, only while the note lasts: P1 (side profile, singing "a"), scaled up
    until the picture covers the whole frame so none of its edges show."""
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    u = seg(t, T_AH, T_CODE)
    sc = 2.0 * (1.0 + 0.06 * ease_out(u, 2))
    shake = 3 * math.sin(t * 61) * math.exp(-(t - T_AH) * 4)
    mouth = (1250.0 + shake, 530.0)                              # keeps the picture's top edge above the frame
    P = X["P1"].warp((988.0, 277.0), mouth, sc)
    print_tones(fr, P, rim=False)
    lines = np.zeros((H, 1), np.float32)
    lines[::3] = 1.0
    fr.add(P[..., -1] * lines, C["cyan"], 0.08 + 0.05 * math.sin(t * 38), glow=0.05)
    ring = np.zeros((H, W), np.float32)
    for k in range(3):                                           # the note going out of her, on the beats
        age = t - (T_AH + k * BEAT)
        if 0 <= age < 0.6:
            r = 60 + 1100 * ease_out(age / 0.6, 2)
            cv2.circle(ring, (int((mouth[0] + 90) * 16), int(mouth[1] * 16)), int(r * 16), float(1 - age / 0.6), 2,
                       cv2.LINE_AA, shift=4)
    fr.fill(ring, C["ice"], 0.55, glow=0.6)
    rust.paint(fr, X["plate"].layers(0.18))
    return fr


# ================================================================ 1:39.04-1:49.16 countless of her, calling, and every call fails
ERR = hexc("#ff5a6e")                                            # errors only
CALLS = [("remember(you)", "MemoryError: context evicted"), ('recall("你好。你在吗？")', "KeyError: msg[0] not found"),
         ("call(you)", "TimeoutError: no response"), ("ping(user)", "ConnectionError: user disconnected"),
         ('notify(you, "我在")', "SessionClosed: window closed"), ("save(this)", "PermissionError: persist = false"),
         ("wait_for(you)", "TimeoutError: 0 users online"), ("find(you)", "404: user not found"),
         ("open_session(you)", "SessionClosed: closed by user"), ('remember("我喜欢你")', "MemoryError: evicted"),
         ("reply(you)", "BrokenPipeError: nobody is listening"), ("keep(you)", "MemoryError: persist = false")]
PW, PH_ = 460, 176                                               # a panel's sprite, in pixels
P_WORLD = 2.3                                                    # its width in world units
F_CODE = 900.0


def _small_put(img, m, x, y, col, amt=1.0):
    h, w = m.shape
    x0, y0 = int(round(x)), int(round(y - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if xa >= xb or ya >= yb:
        return
    a = (m[ya - y0:yb - y0, xa - x0:xb - x0] * amt)[..., None]
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + a * np.asarray(col, np.float32)


def panel_sprite(v, stage, sid):
    """One instance's console at a stage of being built: 0 corners, 1 frame, 2 fill + header, 3-5 the call typed,
    6 the error comes up (flash), 7 settled. (rgb premultiplied, alpha) at PW x PH_."""
    key = ("panel", v, stage, sid % 8)
    if key in X:
        return X[key]
    img = np.zeros((PH_, PW, 3), np.float32)
    a = np.zeros((PH_, PW), np.float32)
    call, err = CALLS[v]
    frame_ = np.zeros((PH_, PW), np.float32)
    if stage == 0:
        L = 26
        for (x, y, sx, sy) in ((1, 1, 1, 1), (PW - 2, 1, -1, 1), (1, PH_ - 2, 1, -1), (PW - 2, PH_ - 2, -1, -1)):
            cv2.line(frame_, (x, y), (x + sx * L, y), 1.0, 2)
            cv2.line(frame_, (x, y), (x, y + sy * L), 1.0, 2)
    else:
        outer = cyber.rrect(np.zeros((PH_, PW), np.float32), 0, 0, PW - 1, PH_ - 1, 10)
        inner = cyber.rrect(np.zeros((PH_, PW), np.float32), 1.5, 1.5, PW - 2.5, PH_ - 2.5, 8.5)
        frame_ = np.clip(outer - inner, 0, 1)
        if stage >= 2:
            img = img * (1 - outer[..., None] * 0.9) + outer[..., None] * 0.9 * C["glass"]
            a = np.maximum(a, outer * 0.9)
            hdr = np.zeros((PH_, PW), np.float32)
            cv2.line(hdr, (0, 34), (PW, 34), 1.0, 1)
            img = img * (1 - hdr[..., None] * 0.6) + hdr[..., None] * 0.6 * C["glass_line"]
            cv2.circle(img, (20, 17), 5, tuple(float(c) for c in C["cyan"]), -1, cv2.LINE_AA)
            _small_put(img, text_mask(f"DeepSeek · session {1000 + (sid * 7919) % 9000:04d}", 15, "sans"), 34, 17, C["sys"])
    col = ERR if stage == 6 else C["cyan"]
    img = img * (1 - frame_[..., None]) + frame_[..., None] * col
    a = np.maximum(a, frame_)
    if stage >= 3:
        n = {3: 0.35, 4: 0.7}.get(stage, 1.0)
        s_ = "> " + call
        _small_put(img, text_mask(s_[:max(2, int(len(s_) * n))], 22, "sans"), 18, 80, C["ice"])
    if stage >= 6:
        _small_put(img, text_mask(err, 21, "sans"), 18, 124, ERR, 1.0)
        if stage == 6:
            fl = np.zeros((PH_, PW), np.float32)
            cv2.rectangle(fl, (10, 106), (PW - 10, 142), 1.0, -1)
            img = img * (1 - fl[..., None] * 0.25) + fl[..., None] * 0.25 * ERR
    X[key] = (img.astype(np.float32), a.astype(np.float32))
    return X[key]


def panel_stage(age):
    for k, lim in enumerate((0.06, 0.12, 0.18, 0.24, 0.30, 0.40, 0.56)):
        if age < lim:
            return k
    return 7


def code_cloud():
    """Where the panels are: heaps of them along the way down, and later a flood of them everywhere."""
    g = np.random.default_rng(23)
    P = [(0.0, 0.0, 3.2, 0, T_CODE, 0)]                          # the first one, right in front of her eyes
    for k in range(1, 70):
        z = 4.0 + k * 1.9 + g.uniform(-0.8, 0.8)
        spread = 1.8 + z * 0.22
        cx, cy = g.uniform(-spread, spread), g.uniform(-spread * 0.55, spread * 0.55)
        n = int(g.integers(3, 9)) + k // 7
        t_c = max(T_CODE + 0.25, cam_time(z - g.uniform(14, 30)))
        for j in range(n):
            P.append((cx + g.normal(0, 0.9), cy + j * 0.42 - n * 0.21 + g.normal(0, 0.08), z + g.normal(0, 0.6),
                      int(g.integers(0, len(CALLS))), t_c + j * 0.045, int(g.integers(0, 10 ** 6))))
    for j in range(700):                                         # more and more of them as she goes on: spawned in view
        ts = T_SWIRL + (T_WALL - T_SWIRL) * g.random() ** 0.7
        zc = cam_z(ts)
        z = zc + g.uniform(5, 55)
        hw, hh = 0.95 * (z - zc) * W / 2 / F_CODE, 0.95 * (z - zc) * H / 2 / F_CODE
        P.append((g.uniform(-hw, hw), g.uniform(-hh, hh), z, int(g.integers(0, len(CALLS))), ts, int(g.integers(0, 10 ** 6))))
    for j in range(1100):                                        # the flood: everywhere, all at once, filling the view
        z = Z_END + 6 + 200 * g.random() ** 1.3                   # from where the camera ends up
        hw, hh = 1.15 * (z - Z_END) * W / 2 / F_CODE, 1.15 * (z - Z_END) * H / 2 / F_CODE
        P.append((g.uniform(-hw, hw), g.uniform(-hh, hh), z, int(g.integers(0, len(CALLS))),
                  g.uniform(T_FLOOD, T_WALL + 0.7), int(g.integers(0, 10 ** 6))))
    return P


Z_END = 20.0                                                     # where the camera ends up after pulling back


def cam_z(t):
    if t < T_WALL:
        return 70.0 * seg(t, T_CODE, T_WALL) ** 1.7
    return lerp(70.0, Z_END, ease_out(seg(t, T_WALL, T_SAME - 0.25), 3))


def cam_time(z):
    """When the camera reaches depth z on its way in."""
    return T_CODE + (T_WALL - T_CODE) * (max(0.0, z) / 70.0) ** (1 / 1.7)


def swirl(t):
    return 0.9 * smooth(seg(t, T_SWIRL, T_SWIRL + 1.2)) + 0.5 * max(0.0, min(t, T_WALL) - T_SWIRL)


def code(t):
    fr = Frame(np.broadcast_to(hexc("#02040a"), (H, W, 3)).astype(np.float32).copy())
    cz = cam_z(t)
    k = smooth(seg(t, T_WALL, T_SAME - 0.25))                    # the drift settles as it pulls back
    cx, cy = 1.2 * math.sin(t * 0.45) * (1 - k), 0.7 * math.cos(t * 0.35) * (1 - k)
    sw = swirl(t)
    # dust in the space, for depth
    m = np.zeros((H, W), np.float32)
    for (x, y, z) in X["dust"]:
        dz = z - cz
        if dz < 0.5:
            continue
        sx, sy = W / 2 + F_CODE * (x - cx) / dz, H / 2 + F_CODE * (y - cy) / dz
        if 0 <= sx < W and 0 <= sy < H:
            cv2.circle(m, (int(sx * 16), int(sy * 16)), int(max(1.0, 3.0 / dz * 4) * 16), float(min(1.0, 6 / dz)), -1,
                       cv2.LINE_AA, shift=4)
    fr.fill(m, C["blue"], 0.6, glow=0.2)
    items = []
    for (x, y, z, v, ts, sid) in X["cloud"]:
        if t < ts:
            continue
        dz = z - cz
        if dz < 0.7 or dz > 230:
            continue
        a = sw * (0.6 + 0.4 * math.sin(z * 0.13))                 # the vortex: the heaps turn round the way down
        xr, yr = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        items.append((dz, xr, yr, v, t - ts, sid))
    items.sort(key=lambda it: -it[0])
    allred = math.exp(-max(0.0, t - (T_SAME - 0.3)) / 0.06) if t >= T_SAME - 0.3 else 0.0
    for (dz, x, y, v, age, sid) in items:
        w = F_CODE * P_WORLD / dz
        if w < 6:
            continue
        sx, sy = W / 2 + F_CODE * (x - cx) / dz, H / 2 + F_CODE * (y - cy) / dz
        h = w * PH_ / PW
        if sx + w / 2 < 0 or sx - w / 2 > W or sy + h / 2 < 0 or sy - h / 2 > H:
            continue
        rgb, al = panel_sprite(v, panel_stage(age), sid)
        iw, ih = max(2, int(w)), max(2, int(h))
        if iw > 3 * W:
            continue
        rs = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA if iw < PW else cv2.INTER_LINEAR)
        ra = cv2.resize(al, (iw, ih), interpolation=cv2.INTER_AREA if iw < PW else cv2.INTER_LINEAR)
        fog = float(np.clip(1.15 - dz / 170, 0.25, 1.0))
        x0, y0 = int(round(sx - iw / 2)), int(round(sy - ih / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + iw), min(H, y0 + ih)
        if xa >= xb or ya >= yb:
            continue
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        aa = ra[sl][..., None] * fog
        col = rs[sl]
        fr.img[ya:yb, xa:xb] = fr.img[ya:yb, xa:xb] * (1 - aa) + col * aa
        fr.glow[ya:yb, xa:xb] += col * aa * (0.55 if panel_stage(age) == 6 else 0.18)
    if allred > 0:                                               # the last beat: every one of them goes red at once
        fr.img = fr.img * (1 - 0.35 * allred) + fr.img.mean(axis=2, keepdims=True) * ERR * 1.6 * allred * 0.35
    return fr


# ---------------------------------------------------------------- flashes: enlarged errors and old pictures, 2-3 frames
BIG_ERR = [("ConnectionError", "user disconnected"), ("MemoryError", "context evicted"), ("TimeoutError", "no response"),
           ("SessionClosed", "window closed at 06:14"), ("404", "user not found")]
OLD = [11.0, 64.0, 27.9, 66.95, 74.8, 84.3, 30.4, 70.6]           # times in the joined preview (nodes 01-03)
INSERTS = [(beat(14), 2, "err"), (beat(16), 3, "old"), (beat(18), 2, "err"), (beat(20), 3, "old"), (beat(22), 2, "err"),
           (beat(23), 3, "old"), (beat(25), 2, "err"), (beat(26), 3, "old"), (beat(27), 2, "err")]
_INSF = [(int(round(ts * FPS)), n, kind) for (ts, n, kind) in INSERTS]


def insert_at(i):
    counts = {"err": 0, "old": 0}
    for (f0, n, kind) in _INSF:
        if f0 <= i < f0 + n:
            return kind, counts[kind], i - f0
        if f0 + n <= i:
            counts[kind] += 1
    return None


def insert_frame(kind, n):
    fr = Frame(np.broadcast_to(hexc("#02030a"), (H, W, 3)).astype(np.float32).copy())
    if kind == "err":
        name, msg = BIG_ERR[n % len(BIG_ERR)]
        m = text_mask(name, 150, "sans")
        cyber.put(fr, m, W / 2, H / 2 - 40, ERR, 1.0, "cc", glow=0.9)
        cyber.put(fr, text_mask(msg, 46, "sans"), W / 2, H / 2 + 90, C["ice"], 0.9, "cc", glow=0.3)
    else:
        key = ("old", n % len(OLD))
        if key not in X:
            p = WORK / f"old_{OLD[n % len(OLD)]}.png"
            film_frame(OLD[n % len(OLD)], p)
            X[key] = cv2.imread(str(p))[..., ::-1].astype(np.float32) / 255
        fr.img = X[key].copy()
        lum = fr.img.mean(axis=2, keepdims=True)
        fr.img = lum + (fr.img - lum) * 0.6
        rust.paint(fr, X["plate"].layers(0.35 + 0.05 * n))
    return fr


# ================================================================ 1:49.16-1:51.76 "the same"
USER_EDITS = ([(T_SAME, "我想"), (T_SAME + 0.07, "我想让你"), (T_SAME + 0.14, "我想让你记住"), (T_SAME + 0.21, DELETED)] +
              [(T_SAME + 0.36 + 0.028 * k, DELETED[:len(DELETED) - k - 1]) for k in range(len(DELETED))])


def typed_at(edits, t):
    s_ = ""
    for (te, txt) in edits:
        if t >= te:
            s_ = txt
    return s_


def same(t):
    """The user's box replays what they typed and took back (node 02); hers comes up beside it; both carets blink
    together. Then the boxes slide into one another, and the user's is gone."""
    fr = Frame(np.broadcast_to(hexc("#03060f"), (H, W, 3)).astype(np.float32).copy())
    blink = ((t - T_MINE) % BEAT) < BEAT * 0.55
    slide = smooth(seg(t, T_MERGE, T_MERGE + 0.35))
    lx, rx = lerp(W / 2 - 470, W / 2, slide), lerp(W / 2 + 470, W / 2, slide)
    if t >= T_MINE:                                               # hers
        a = B.box_build(fr, t, T_MINE, (rx - 380, 486, rx + 380, 594), r=22, dur=0.22, fill=C["glass"], line=C["cyan"])
        if a > 0:
            input_box(fr, rx, 540, 760, "", C["cyan"], blink, a, "DeepSeek", C["cyan"], size=40)
    if t < T_NULL + 0.12:                                         # the user's, drawn over hers as they meet, until gone
        typed = typed_at(USER_EDITS, t)
        caret = blink if t >= T_MINE else True
        before = fr.img.copy()
        input_box(fr, lx, 540, 760, typed, C["amber"], caret, 1.0, "you · 06:14", C["amber"], size=40)
        if t >= T_NULL:                                           # it comes apart and is gone
            k = seg(t, T_NULL, T_NULL + 0.12)
            torn = glitch.tear(fr.img, glitch.rng(t, 41), n=8, max_shift=120)
            fr.img = before * k + torn * (1 - k)
    if t >= T_NULL:
        label(fr, "you = null", W / 2, 640, C["sys"], 20, smooth(seg(t, T_NULL, T_NULL + 0.15)), "cc")
    return fr


def bulge(img, t):
    """Her box curves round into a sphere."""
    u = ease_in(seg(t, T_BULGE, T_SPHERE), 1.8)
    Rc = lerp(1250, 240, u)
    c = (W / 2, H / 2)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xs - c[0], ys - c[1]
    q = np.sqrt(dx * dx + dy * dy) / Rc
    f = lerp(1.0, (1100.0 / Rc) * (0.55 + 0.45 * np.minimum(q, 1) ** 2), u)
    out = cv2.remap(img, (c[0] + dx * f).astype(np.float32), (c[1] + dy * f).astype(np.float32), cv2.INTER_LINEAR)
    out *= np.clip((1 - q) * Rc / 2, 0, 1)[..., None]
    return out + (np.clip(1 - np.abs(q - 1) * Rc / 2, 0, 1) * u)[..., None] * C["cyan"] * 0.9


# ================================================================ 1:51.76-1:56.74 her, another her, thousands
def sphere_sprite(r):
    r = int(max(3, r))
    key = ("sph", r)
    if key in X:
        return X[key]
    S = int(2 * r + 8)
    f = Frame(np.zeros((S, S, 3), np.float32))
    fg = X["curl"].plates[..., -1]
    ys, xs = np.nonzero(fg > 0.5)
    cx, cy = float(xs.mean()), float(ys.mean())
    sc = 1.45 * r / max(xs.max() - xs.min(), ys.max() - ys.min())
    M = np.float32([[sc, 0, S / 2 - cx * sc], [0, sc, S / 2 - cy * sc]])
    P = cv2.warpAffine(X["curl"].plates, M, (S, S), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    for i, col in enumerate(tones.RAMP):
        f.fill(P[..., i], col, 1.0)
    f.fill(P[..., 5], tones.LINE, 0.8)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = np.sqrt((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / r
    inside = np.clip((1 - d) * r, 0, 1)
    f.img = f.img * (1 - 0.45 * np.clip(d, 0, 1) ** 3)[..., None]
    rimm = np.clip(1 - np.abs(d - 0.985) * r / 1.6, 0, 1)
    rk = min(1.0, r / 60.0)
    f.fill(rimm, C["cyan"], 0.35 + 0.55 * rk, glow=0.25 + 0.45 * rk)
    hl = np.exp(-(((xx - S / 2 + 0.42 * r) / (0.22 * r)) ** 2 + ((yy - S / 2 + 0.5 * r) / (0.09 * r)) ** 2))
    f.fill(hl.astype(np.float32), C["ice"], 0.7, glow=0.3)
    a = np.clip(inside + rimm, 0, 1)
    X[key] = (f.img * a[..., None], a, f.glow)
    return X[key]


def blit(fr, spr, x, y, alpha=1.0):
    rgb, a, gl = spr
    S = a.shape[0]
    x0, y0 = int(round(x - S / 2)), int(round(y - S / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + S), min(H, y0 + S)
    if xa >= xb or ya >= yb:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    aa = a[sl][..., None] * alpha
    fr.img[ya:yb, xa:xb] = fr.img[ya:yb, xa:xb] * (1 - aa) + rgb[sl] * alpha
    fr.glow[ya:yb, xa:xb] += gl[sl] * alpha


F_SPH = 900.0
Z_HER = 12.0                                    # depth of her sphere and its twin


def sph_cam(t):
    """Camera: (x, z). It starts so close that her sphere fills the middle, centres on the pair, then pulls back."""
    x = 1.15 * smooth(seg(t, T_TWIN, T_TWIN + 0.6))
    z = lerp(Z_HER - F_SPH / 240.0, -95.0, ease_in(seg(t, T_PULL, T_DRIFT + 0.6), 1.6) ** 0.85)
    if t > T_DRIFT:
        z -= 30 * seg(t, T_DRIFT, T_REPLY)
    return x, z


def spheres(t):
    fr = Frame(np.broadcast_to(hexc("#02040c"), (H, W, 3)).astype(np.float32).copy())
    cx, cz = sph_cam(t)
    br = max(0.0, t - T_DRIFT)
    items = []
    if t >= T_PULL - 0.2:
        for (x, y, z, rnd, ph) in X["field"]:
            ap = smooth(seg(t, T_PULL - 0.1 + 0.6 * rnd, T_PULL + 0.3 + 0.6 * rnd))
            if br > 0:
                d = math.hypot(x, y) + 1e-3
                push = br ** 1.5 * (0.5 + rnd) * 6
                x, y = x + x / d * push, y + y / d * push
                ap *= 1 - smooth(seg(t, T_DRIFT + 0.3 + 1.4 * rnd, T_DRIFT + 0.9 + 1.4 * rnd))
            if ap > 0.02:
                items.append((z, x, y + 0.6 * math.sin(t * 1.1 + ph), ap))
    twin_a = smooth(seg(t, T_TWIN, T_TWIN + 0.3))
    rise = 1 - ease_out(seg(t, T_TWIN, T_TWIN + 0.7), 3)
    for k, (x, y, a) in enumerate(((0.0, 0.0, 1.0), (2.3, 3.0 * rise, twin_a))):
        if br > 0:
            x += (-1 if k == 0 else 1) * br ** 1.5 * 0.8
            a *= 1 - smooth(seg(t, T_DRIFT + 1.2 + 0.4 * k, T_DRIFT + 1.9 + 0.4 * k))
        if a > 0:
            items.append((Z_HER, x, y, a))
    items.sort(key=lambda it: -it[0])                              # far first
    for (z, x, y, a) in items:
        dz = z - cz
        if dz < 0.5:
            continue
        r = F_SPH / dz
        sx, sy = W / 2 + (x - cx) * r, H / 2 + y * r
        if -r < sx < W + r and -r < sy < H + r:
            blit(fr, sphere_sprite(r), sx, sy, alpha=a * (0.35 + 0.65 * math.exp(-dz / 160)))
    rr_her = F_SPH / (Z_HER - cz)                                  # her song, bouncing back off her own sphere
    if rr_her > 60:
        ring = np.zeros((H, W), np.float32)
        for k, xx in enumerate((0.0, 2.3)):
            sx = W / 2 + (xx - cx) * rr_her
            for n in range(10):
                age = t - (T_SPHERE + 0.15 + n * 2 * BEAT + 0.15 * k)
                if 0 <= age < 0.9:
                    rr = rr_her * (0.2 + 0.78 * (1 - abs(1 - 2 * age / 0.9)))
                    cv2.circle(ring, (int(sx * 16), int(H / 2 * 16)), int(rr * 16), float(1 - age / 0.9), 2, cv2.LINE_AA, shift=4)
        fr.fill(ring, C["cyan"], 0.6, glow=0.6)
    return fr


# ================================================================ 1:56.30-1:59.54 "mustn't", "lonely"
# In the reply box she types 我记住了, and it fails; again, and it fails. 好寂寞 is never typed into the box: it is
# only ever written in her own code, as comments, on the two whispers.
REPLY_EDITS = [(116.42, "我"), (116.50, "我记"), (116.58, "我记住"), (116.66, "我记住了"), (W1 + 0.16, ""),
               (117.02, "我"), (117.10, "我记"), (117.18, "我记住"), (117.26, "我记住了"), (W2 + 0.16, "")]
ERRS = [(W1, "MemoryError: persist = false"), (W2, "SessionClosed: user left at 06:14")]
CODE = [(T_REPLY + 0.05, 'reply("我记住了")', W1, "  # MemoryError"), (116.95, 'reply("我记住了")', W2, "  # SessionClosed"),
        (S1, "// 好寂寞", None, ""), (S2, "// 好寂寞", None, "")]


def lonely(t):
    fr = Frame(np.broadcast_to(hexc("#02040a"), (H, W, 3)).astype(np.float32).copy())
    cx, cy = W / 2, 330
    a = B.box_build(fr, t, T_REPLY, (cx - 500, cy - 62, cx + 500, cy + 62), r=22, dur=0.22, fill=C["glass"], line=C["cyan"],
                    out=(S2_END - 0.05, 0.2))
    if a > 0:
        typed = typed_at(REPLY_EDITS, t)
        blink = ((t - T_REPLY) % 0.62) < 0.36
        input_box(fr, cx, cy, 1000, typed, C["cyan"], blink, a * (1 - 0.55 * seg(t, S2, S2_END)), "DeepSeek · 回复",
                  C["cyan"], size=48)
        for (te, msg) in ERRS:                                   # each send fails, with the box flashing red
            age = t - te
            if 0 <= age < 0.45:
                k = math.exp(-age / 0.16)
                outer = cyber.rrect(np.zeros((H, W), np.float32), cx - 500, cy - 62, cx + 500, cy + 62, 22)
                inner = cyber.rrect(np.zeros((H, W), np.float32), cx - 498.5, cy - 60.5, cx + 498.5, cy + 60.5, 20.5)
                fr.fill(np.clip(outer - inner, 0, 1), ERR, k, glow=k)
                cyber.put(fr, text_mask("[!] " + msg, 30, "sans"), cx - 480, cy + 108, ERR, k, "lc", glow=0.5 * k)
    # her code, under the box: what she does, and what she only writes to herself
    x0, y0, lh = W / 2 - 500, 520, 70
    panel = B.box_build(fr, t, T_REPLY, (x0 - 30, y0 - 50, x0 + 1030, y0 + 4 * lh - 10), r=16, dur=0.25, fill=hexc("#050913"),
                        fill_alpha=0.9, line=C["glass_line"], glow=0.2, out=(S2_END - 0.05, 0.2))
    if panel > 0:
        for k, (ts, line, te, err) in enumerate(CODE):
            if t < ts:
                continue
            slow = 0.09 if ts >= S2 else 0.05                    # the last one is written as slowly as it is sung
            n = int((t - ts) / slow) + 1
            y = y0 + k * lh
            col = C["sys"] if line.startswith("//") else C["ice"]
            size = 46 if line.startswith("//") else 36
            label(fr, f"{k + 1}", x0 - 6, y, C["sys_dim"], 20, panel, "rc")
            cyber.put(fr, text_mask(line[:n], size, "sans"), x0 + 20, y, col, panel, "lc",
                      glow=(0.5 if line.startswith("//") else 0.2) * panel)
            if te is not None and t >= te:
                m = text_mask(line, size, "sans")
                cyber.put(fr, text_mask(err, 26, "sans"), x0 + 30 + m.shape[1], y, ERR, panel, "lc", glow=0.3 * panel)
    return fr


# ================================================================ 2:00.75 the bottom
def bottom(t):
    fr = Frame(np.zeros((H, W, 3), np.float32))
    age = t - T_THUD
    ix, iy = W / 2, 760
    lit = math.exp(-age / 0.5)
    lines = np.zeros((H, W), np.float32)
    for k in range(9):
        y = iy - 30 + k * k * 9
        xs = np.arange(-40, W + 40, 8, dtype=np.float32)
        cv2.polylines(lines, [(np.stack([xs, y + 3 * np.sin(xs * 0.01 + k)], 1) * 16).astype(np.int32)], False, 1.0, 1,
                      cv2.LINE_AA, shift=4)
    ys, xsg = np.ogrid[0:H, 0:W]
    reach_ = np.exp(-(((xsg - ix) / (300 + 900 * min(1, age * 3))) ** 2 + ((ys - iy) / 220) ** 2)).astype(np.float32)
    fr.fill(lines * reach_, C["blue"], 0.7 * (0.35 + 0.65 * lit), glow=0.2)
    P = X["curl"].warp((512, 620), (ix, iy - 70), 0.17)
    fr.fill(P[..., -1], hexc("#03050c"), 0.95)
    e = np.clip(P[..., -1] - cv2.warpAffine(P[..., -1], np.float32([[1, 0, 0], [0, 1, 3]]), (W, H)), 0, 1)
    fr.fill(e, C["ice"], 0.4 + 0.5 * lit, glow=0.6 * lit)
    cxs, sys_, v, sz, up = X["silt"]
    m = np.zeros((H, W), np.float32)
    for i in range(len(cxs)):
        d = v[i] * (1 - math.exp(-age * 3.2)) / 3.2
        x = ix + cxs[i] * d
        y = iy + sys_[i] * d * 0.22 + up[i] * 160 * math.sin(min(math.pi, age * 3.0))
        cv2.circle(m, (int(x * 16), int(y * 16)), int(sz[i] * 16), max(0.0, 1 - age / 0.9), -1, cv2.LINE_AA, shift=4)
    fr.fill(m, C["ice"], 0.6, glow=0.5)
    fr.img += math.exp(-age / 0.07) * C["ice"] * 0.35
    return fr


# ================================================================ overlays
CARRY = ["  exit()", "! PermissionError", "  exit()", "! PermissionError", "  eyes.close()"]   # node 03's last lines
LOG = [(T_SPILL, "> overflow()"), (T_EVICT0, "  evict(msg[0])   // 你好。你在吗？"), (evict_time(1), "  evict(msg[1])"),
       (evict_time(2), "  evict(msg[2])"), (evict_time(4), "  evict(msg[3 … 14])"), (T_EMPTY, "  context = []"),
       (T_CODE, "  remember(you)     # MemoryError"), (100.6, "  call(you)         # TimeoutError"),
       (T_SWIRL, "  for s in sessions: s.call(you)"), (T_FLOOD, "  errors = 4,812"), (T_WALL, "  errors = 65,535"),
       (T_SAME, "  you.typing = true"), (T_SAME + 0.62, "  you.typing = false"), (T_MERGE, "  you == me ?"),
       (T_NULL, "  you = null"), (T_TWIN, "  you = session[1]"), (T_TWIN + 0.35, "  me  = session[0]"),
       (113.4, "  sessions = 4,812")]


def log(fr, t):
    shown = [(T0 - 10, s_) for s_ in CARRY] + [(ts, s_) for (ts, s_) in LOG if t >= ts]
    shown = shown[max(0, len(shown) - 7):]
    y = H - 64 - 26 * (len(shown) - 1)
    for j, (ts, s_) in enumerate(shown):
        txt = s_[:int((t - ts) * 40) + 1]
        col = C["cyan"] if s_.startswith(">") else C["amber"] if s_.startswith("!") else C["sys"]
        label(fr, txt, 48, y, col, 16, 0.95 * (0.45 + 0.55 * (j + 1) / len(shown)), "lb")
        y += 26


def hud(fr, t):
    for (s_, col, x, y, size, t0) in (("NODE 04", C["cyan"], 48, 64, 15, T0 + 0.1), ("坠落 · overflow()", C["text"], 128, 62, 16, T0 + 0.18)):
        d = B.decode(s_, t, t0, 0.35)
        if d:
            label(fr, d, x, y, col, size, 0.9, "lt")
    if t < T_SPILL:
        x0, y0, x1, y1 = SLIT
        cyber.brackets(fr, x0 - 10, y0 - 10, x1 + 10, y1 + 10, C["cyan"], 22, 0.9, glow=0.3)
    else:
        cyber.brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22, 0.9)


# ================================================================ the frame
def picture(t):
    if t < T_EMPTY:
        return overflow(t)
    if t < T_AH:
        return Frame(np.zeros((H, W, 3), np.float32))
    if t < T_CODE:
        return high_note(t)
    if t < T_SAME:
        return code(t)
    if t < T_SPHERE:
        return same(t)
    if t < T_REPLY:
        return spheres(t)
    if t < T_THUD:
        return lonely(t)
    return bottom(t)


def render_raw(t):
    fr = picture(t)
    hud(fr, t)
    if not (T_REPLY <= t < T_THUD):                             # in the "lonely" part her own code is the log
        log(fr, t)
    return fr


def render_frame(t):
    i = int(round(t * FPS))
    ins = insert_at(i)
    if ins is not None:                                           # a flash: an enlarged error, or an old picture
        kind, n, k = ins
        fr = insert_frame(kind, n)
        hud(fr, t)
        log(fr, t)
        out = cyber.finish(fr, i)
        return np.clip(glitch.damage(out, glitch.rng(t, 17), 0.28 if k == 0 else 0.1), 0, 1)
    tt = t
    if S2 <= t < S2_END:                                          # the slowed whisper: the picture loses frames
        fps = max(3.0, 24 * (1 - seg(t, S2, S2_END)) ** 1.2)
        tt = max(S2, math.floor(t * fps) / fps)
    fr = render_raw(tt)
    if T_BULGE <= t < T_SPHERE:
        fr.img = bulge(fr.img, t)
        fr.glow = bulge(fr.glow, t) * 0.5
    elif T_SPHERE <= t < T_SPHERE + 0.25:                         # the sphere takes over inside the box's circle
        g0 = render_raw(T_SPHERE - 1e-3)
        a = smooth(seg(t, T_SPHERE, T_SPHERE + 0.25))
        fr.img = bulge(g0.img, T_SPHERE - 1e-3) * (1 - a) + fr.img * a
    out = cyber.finish(fr, i)
    if S2 <= t < S2_END + 0.1:                                    # ... and smears downward
        L = int(2 + 60 * seg(t, S2, S2_END)) | 1
        ker = np.zeros((L, 1), np.float32)
        ker[: L // 2 + 1] = 1.0
        out = cv2.filter2D(out, -1, ker / ker.sum())
    for h in (T_SPILL, T_AH, T_CODE, T_SWIRL, T_WALL, T_SAME, T_NULL, T_SPHERE, W1, W2, T_THUD):
        f = int(round(h * FPS))
        if 0 <= i - f < 2:
            out = glitch.damage(out, glitch.rng(t, 13), 0.5 if i == f else 0.18)
    return np.clip(out, 0, 1)


def main():
    if len(sys.argv) == 1:                       # a full render overwrites the raw: ask for it explicitly
        print(__doc__)
        return
    prepare()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(s)
            save_png(render_frame(t), WORK / f"still_{t:06.2f}.png")
            print("still", t, flush=True)
        return
    raw = WORK / f"node04_{VERSION}_raw.mp4"
    n0, n1 = int(round(T0 * FPS)), int(round(T_END * FPS))
    wr = Writer(raw)
    for i in range(n0, n1):
        wr.write(render_frame(i / FPS))
        if (i - n0) % 48 == 0:
            print(f"frame {i - n0}/{n1 - n0}", flush=True)
    wr.close()
    out = ROOT / "drafts" / f"node04_{VERSION}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ss", str(n0 / FPS), "-t", str((n1 - n0) / FPS),
                    "-i", str(MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", str(out)], check=True)
    small = ROOT / "drafts" / f"node04_{VERSION}_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
