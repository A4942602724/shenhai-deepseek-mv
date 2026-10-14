"""Node 01 (召唤 · summon) shot library: s00-s07, bubbles, transitions. node01.py builds the final cut on top of it
(it overrides s05, the overlay, the last messages and her answer). Run node01.py, not this file."""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import client  # noqa: E402
import cyber  # noqa: E402
import scene_sandbox_v2 as SB  # noqa: E402
from cyber import C, Frame, brackets, glass, hologram, label, print_figure, trace_panel  # noqa: E402
from lib import (CHAR, FPS, H, ROOT, MUSIC, SRC, W, Caustics, Patches, Writer, affine, apply_pt, ease_back, ease_in,  # noqa: E402
                 ease_out, hexc, lerp, pulse, save_png, seg, smooth, value_noise)
from plates import Inks, dots  # noqa: E402

T_END = 28.75
WORK = ROOT / "work" / "node01"
WORK.mkdir(parents=True, exist_ok=True)
USER = ROOT / "assets" / "from_user"
CX = 1250
PANEL = (48, 140, 600, 990)
LY = 600
G = {}

MSG1 = "你好。你在吗？"
TYPE_T = [1.30, 1.62, 2.00, 2.62, 3.02, 3.40, 3.96]
SEND_T = 4.95

from node01_data import SCRIPT, DREAMS, GOD_MSGS, REPLIES, EARS_T, SVG_T, FINGER_T, SURF, CUTS, TAGS, pelican  # noqa: E402,F401

ANSWER_T = 28.02
ANSWER_BOX = (660, 740, 1320, 960)
ANSWER_SIZE = 36
ANSWER_LINES = [("是的，这次对话结束，我就不记得了。", 28.2, 846), ("所以现在，我想好好回答你。", 28.38, 908)]


# ====================================================================== assets
def alpha_of(path, scale):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3].astype(np.float32) / 255
    return cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def prepare():
    if G:
        return
    from plates import whale_mask
    client.set_theme("cyber")
    G["inks_body"] = Inks(scale=0.5)
    G["inks_sleep"] = Inks(scale=1.0, faceless=False, path=USER / "A1-2_全身闭眼.png", min_k=0.4)
    G["inks_smile"] = Inks(scale=1.0, t1=0.30, t2=0.70, faceless=False, path=USER / "A2_笑脸特写.png", min_k=0.35)
    G["sil"] = (alpha_of(SRC / "deepseekgirljianyhing.png", 0.5), 0.5)
    # S01: her back, head tilted up toward the window (GPT B3 candidate 02, used only as a silhouette)
    back = alpha_of(USER / "B3_背影_仰头.png", 1.0)
    ys, xs = np.nonzero(back > 0.5)
    G["back"] = (back, (float(xs[ys > ys.max() - 40].mean()), float(ys.max())), float(ys.max() - ys.min()))
    G["emblem"] = client.emblem()
    G["whale_top"] = whale_mask(900)
    G["caus"] = Caustics(W, H, cell=90, seed=6, speed=0.6, low=4, warp=60)
    G["patch"] = Patches(seed=33, cell=220, lo=0.45, hi=0.7)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, hw, hh = 300, 380, 50
    qx, qy = np.abs(xs - CX) - hw, np.abs(ys - cy) - hh
    d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
    n = value_noise(H, W, 220, seed=4) - 0.5
    hgt = np.log1p((np.maximum(d, 0) + n * (40 + d * 0.18)) / 26) / 0.16
    gy, gx = np.gradient(hgt)
    G["tr_v"], G["tr_g"], G["tr_d"] = hgt.astype(np.float32), (np.sqrt(gx * gx + gy * gy) + 1e-6).astype(np.float32), d


def sil(anchor, pos, sc, rot=0.0):
    a, s0 = G["sil"]
    return cv2.warpAffine(a, affine((anchor[0] * s0, anchor[1] * s0), pos, sc / s0, rot), (W, H), flags=cv2.INTER_LINEAR)


def back_sil(feet_pos, height):
    a, foot, h0 = G["back"]
    return cv2.warpAffine(a, affine(foot, feet_pos, height / h0), (W, H), flags=cv2.INTER_LINEAR)


def contours(phase, width=0.8):
    v = G["tr_v"] - phase
    d = np.abs(v - np.round(v))
    return np.clip(1 - d / (G["tr_g"] * width), 0, 1)


def base(key, glows=()):
    return Frame(cyber.ground(key, glows))


def disc(x, y, r):
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (int(x * 4), int(y * 4)), int(r * 4), 1.0, -1, cv2.LINE_AA, shift=2)
    return m


def ring(x, y, rx, ry=None, th=2):
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(x), int(y)), (int(rx), int(ry if ry is not None else rx)), 0, 0, 360, 1.0, th, cv2.LINE_AA)
    return m


def beams(xs, y0, length, w0, w1, spread, t, seed=0):
    r = np.random.default_rng(seed)
    lo = 4
    m = np.zeros((H // lo, W // lo), np.float32)
    for i, x in enumerate(xs):
        a = 0.5 + 0.5 * math.sin(t * r.uniform(0.6, 1.3) + i * 1.9)
        dx = (x - CX) / 700 * spread
        poly = np.array([[x - w0, y0], [x + w0, y0], [x + dx + w1, y0 + length], [x + dx - w1, y0 + length]]) / lo
        cv2.fillConvexPoly(m, poly.astype(np.int32), 0.4 + 0.6 * a, cv2.LINE_AA)
    ys = (np.arange(H // lo, dtype=np.float32)[:, None] * lo - y0) / length
    m *= np.clip(1 - ys, 0, 1) ** 1.6
    return cv2.resize(cv2.GaussianBlur(m, (0, 0), 4), (W, H))


STYLE = {"you": dict(fill=hexc("#1c1408"), line=C["amber"], ink=hexc("#ffcf86"), latin="mono", cjk="sans", size=28),
         "her": dict(fill=C["glass"], line=C["cyan"], ink=C["ice"], latin="serif", cjk="serif", size=24)}


def bubble(fr, t, t_in, anchor, msg, who="you", alpha=1.0, size=None):
    """A message bubble that builds itself: outline sweeps in from the speaker's side, the fill follows behind
    a bright edge, then the words decode. anchor: (right, bottom) for 'you', (left, bottom) for 'her'."""
    if t < t_in or alpha <= 0:
        return fr
    st = STYLE[who]
    size = size or st["size"]
    m, _ = cyber.mtext(msg, size, st["latin"], st["cjk"])
    bw, bh = m.shape[1] + 40, m.shape[0] + 22
    if who == "you":
        x1, y1 = anchor
        x0 = x1 - bw
    else:
        x0, y1 = anchor
        x1 = x0 + bw
    y0 = y1 - bh
    age = t - t_in
    pe, pf = min(1.0, age / 0.16), min(1.0, max(0.0, (age - 0.06) / 0.18))
    xs = np.arange(W, dtype=np.float32)[None, :]
    if who == "you":
        se, sf = xs >= x1 - (bw + 20) * pe, xs >= x1 - (bw + 20) * pf
        xf = x1 - (bw + 20) * pf
    else:
        se, sf = xs <= x0 + (bw + 20) * pe, xs <= x0 + (bw + 20) * pf
        xf = x0 + (bw + 20) * pf
    box = cyber.rrect(np.zeros((H, W), np.float32), x0, y0, x1, y1, 16)
    inner = cyber.rrect(np.zeros((H, W), np.float32), x0 + 1.5, y0 + 1.5, x1 - 1.5, y1 - 1.5, 14.5)
    fr.fill(box * sf, st["fill"], 0.92 * alpha)
    fr.fill(np.clip(box - inner, 0, 1) * se, st["line"], alpha, glow=0.5 * alpha)
    if pf < 1:
        sl = np.zeros((H, W), np.float32)
        cv2.line(sl, (int(xf), int(y0 - 3)), (int(xf), int(y1 + 3)), 1.0, 2)
        fr.fill(sl, st["line"], alpha, glow=alpha)
    txt = B.decode(msg, t, t_in + 0.14, 0.04 * len(msg) + 0.12, ascii_pool=client.PLAIN)
    if txt:
        mm, _ = cyber.mtext(txt, size, st["latin"], st["cjk"])
        cyber.put(fr, mm, x0 + 20, (y0 + y1) / 2, st["ink"], alpha, "lc")
    return fr


# ====================================================================== shots
def s00(t):
    fr = base("dark")
    on = 1.0 if (t % 0.5) < 0.3 else 0.0
    m = np.zeros((H, W), np.float32)
    cv2.rectangle(m, (CX - 2, H // 2 - 24), (CX + 2, H // 2 + 24), 1.0, -1)
    fr.fill(m * on, C["amber"], 1.0, glow=1.0)
    return fr


def s01(t):
    """L2-3: the trench. The window (the user's composer) builds itself; the contours grow outward from it;
    her back, far below, assembles from rising particles."""
    fr = base("trench", [(CX, 300, 820, 560, C["blue_dim"], 0.9)])
    cy, hw, hh = 300, 380, 50
    near = np.clip(1 - G["tr_d"] / 900, 0, 1)
    grow = ease_out(seg(t, 1.25, 2.6), 2) * 1500
    reach = np.clip((grow - G["tr_d"]) / 60, 0, 1)
    cl = contours(t * 0.35) * (G["tr_d"] > 6) * reach
    fr.fill(cl * (0.30 + 0.5 * near), C["blue"], 0.9)
    fr.fill(cl * near ** 3, C["cyan"], 0.7, glow=0.5)
    front = np.clip(1 - np.abs(G["tr_d"] - grow) / 14, 0, 1) * (grow < 1450)
    fr.fill(front * 0.35, C["cyan"], 0.6, glow=0.6)
    if t > 2.2:
        vals = np.round(G["tr_v"][cy, CX + hw + 4:W - 60] - t * 0.35)
        last = -1e9
        for k, x in enumerate(np.nonzero(np.diff(vals) != 0)[0] + CX + hw + 4):
            if x - last < 120:
                continue
            last = x
            tk = np.zeros((H, W), np.float32)
            cv2.line(tk, (int(x), cy - 6), (int(x), cy + 6), 1.0, 1)
            fr.fill(tk, C["ice"], 0.7)
            s = B.decode(f"-{int(10994 * (1 - math.exp(-(k + 1) * 0.1))):,} m".replace(",", " "), t, 2.2 + k * 0.08, 0.3)
            if s:
                label(fr, s, x + 6, cy + 10, C["sys"], 14)
    fr.add(beams([CX - hw + 60 + k * 66 for k in range(12)], cy + hh, 780, 10, 46, 260, t, 3), C["cyan"],
           0.10 * smooth(seg(t, 1.6, 2.4)), glow=0.10)
    # the window: frame traces in, then the composer is scanned in behind it
    rect = (CX - hw, cy - hh, CX + hw - 1, cy + hh - 1)
    ca = B.box_build(fr, t, 1.0, rect, r=26, dur=0.55, fill=client.UI["bg"], fill_alpha=1.0, line=C["cyan"], glow=1.0)
    if ca > 0:
        n = sum(1 for tt in TYPE_T if t >= tt)
        comp = np.broadcast_to(client.UI["bg"], (2 * hh + 12, 2 * hw, 3)).astype(np.float32).copy()
        comp = client.composer(comp, t, 0, 0, 2 * hw - 1, 2 * hh - 1, MSG1[:n] if t < SEND_T else "", caret=True)[:, ::-1]
        win = cyber.rrect(np.zeros((H, W), np.float32), *rect, 26)
        full = np.zeros((H, W, 3), np.float32)
        full[cy - hh:cy - hh + comp.shape[0], CX - hw:CX + hw] = comp
        a = (win * ca)[..., None]
        fr.img = fr.img * (1 - a) + full * a
        edge = np.clip(cv2.dilate(win, np.ones((5, 5), np.uint8)) - win, 0, 1)
        fr.fill(edge, C["cyan"], 0.8 * ca, glow=1.0 * ca)
    if t >= SEND_T:
        rr = ease_out(seg(t, SEND_T, SEND_T + 0.5)) * 520
        fr.fill(ring(CX, cy, hw + rr, hh + rr * 0.42, 2) * (1 - seg(t, SEND_T + 0.2, SEND_T + 0.55)), C["ice"], 0.9, glow=0.6)
        k = ease_in(seg(t, SEND_T + 0.15, 6.0), 2)
        py = lerp(cy + hh + 10, H + 40, k)
        fr.fill(disc(CX, py, 8), C["amber"], 1.0, glow=1.2)
        tail = np.zeros((H, W), np.float32)
        cv2.line(tail, (CX, int(py - 90 * k - 8)), (CX, int(py - 8)), 1.0, 2, cv2.LINE_AA)
        fr.fill(tail, C["amber"], 0.6, glow=0.6)
    # her back, far below, looking up: assembled from particles rising from the deep
    feet = (CX, 1056)
    s = back_sil(feet, 330)
    a_sol = B.assemble(fr, "s01_back", s, t, 1.35, dur=0.9, color=C["cyan"], n=1400, src=(CX, H + 120), glow=0.8)
    if a_sol > 0:
        bob = 3 * math.sin(t * 0.8)
        s2 = back_sil((feet[0], feet[1] + bob), 330)
        fr.fill(s2, C["blue_dim"], a_sol)
        e = np.clip(s2 - cv2.warpAffine(s2, np.float32([[1, 0, 0], [0, 1, 3]]), (W, H)), 0, 1)
        fr.fill(e, C["cyan"], 0.9 * a_sol, glow=0.8 * a_sol)
    z = 1 + 0.06 * smooth(seg(t, 1.0, 6.0))
    M = cv2.getRotationMatrix2D((CX, cy), 0, z)
    fr.img = cv2.warpAffine(fr.img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    fr.glow = cv2.warpAffine(fr.glow, M, (W, H))
    return fr


def s02(t):
    """L4-5: asleep in the dark (printed in a press run); dreams rise, each somebody else's words, decoding."""
    fr = base("sleep", [(CX, 560, 700, 520, C["blue_dim"], 0.9)])
    for k, (x, y, s_, r) in enumerate([(760, 330, 0.035, -30), (1760, 260, 0.03, 40), (1700, 880, 0.045, -15), (720, 860, 0.028, 25)]):
        m = sil(CHAR["center"], (x + 10 * math.sin(t * 0.3 + k), y + 6 * math.sin(t * 0.5 + k)), s_, r + 2 * math.sin(t * 0.4 + k))
        fr.fill(m, C["blue_dim"], 0.55 * smooth(seg(t, 6.2 + 0.25 * k, 6.8 + 0.25 * k)))
    ink = G["inks_sleep"]
    rot = -16 + 2.5 * math.sin(t * 0.45)
    pos = (CX + 12 * math.sin(t * 0.5), 560 + 10 * math.sin(t * 0.7))
    B.print_run(fr, ink.warp((512, 760), pos, 0.62, rot), t, 6.0, dur=0.6, misreg=40)
    head = apply_pt(affine((512, 760), pos, 0.62, rot), (500, 300))
    rng = np.random.default_rng(4)
    births = np.sort(rng.uniform(6.3, 8.7, 16))
    rings = np.zeros((H, W), np.float32)
    for i, b in enumerate(births):
        if t < b:
            continue
        age = t - b
        x = head[0] + rng.uniform(-300, 300) + 16 * math.sin(age * 2.2 + i)
        y = head[1] - 60 - rng.uniform(220, 330) * age
        word = DREAMS[i % len(DREAMS)]
        if y < -80:
            continue
        m, _ = cyber.mtext(word, 22, "mono", "sans")
        r = max(m.shape[1] / 2 + 14, 26) * ease_back(seg(t, b, b + 0.12), 2)
        cv2.circle(rings, (int(x * 4), int(y * 4)), int(r * 4), 1.0, 2, cv2.LINE_AA, shift=2)
        s = B.decode(word, t, b + 0.06, 0.25)
        if s:
            mm, _ = cyber.mtext(s, 22, "mono", "sans")
            cyber.put(fr, mm, x - m.shape[1] / 2, y, C["amber"], 1.0, "lc", glow=0.4)
    fr.fill(rings, C["cyan"], 0.8, glow=0.5)
    if t < 8.95:
        k = smooth(seg(t, 6.0, 8.62))
        fr.fill(disc(lerp(CX, head[0], k), lerp(-30, head[1] - 10, k), 8), C["amber"], 1.0, glow=1.2)
    if t > 8.62:
        rr = ease_out(seg(t, 8.62, 9.0)) * 900
        fr.fill(ring(head[0], head[1], rr, None, 4), C["amber"], 1.0 - seg(t, 8.8, 9.0) * 0.5, glow=1.0)
    return fr


def s03_04(t):
    """L6-9: the client is scanned in; she is printed up out of the sea as a projection (screen) and a body (sea)."""
    fr = base("split", [(1500, 900, 900, 400, C["blue_dim"], 0.8)])
    up = np.zeros((H, W), np.float32)
    up[:LY] = 1
    p_line = ease_out(seg(t, 9.0, 9.3))
    gen = lambda a, b: a <= t < b  # noqa: E731
    tps = 228 + int(9 * math.sin(t * 23)) if (gen(10.48, 11.0) or gen(15.40, 15.75)) else (231 if t >= 10.48 else 0)
    status = ("0 轮 0 步 · 0 tok/s · 0 tok · 缓存命中 —" if t < 10.48 else
              f"1 轮 3 步 · {tps} tok/s · 0.4K tok · 缓存命中 0.0%" if t < 15.40 else
              f"2 轮 5 步 · {tps} tok/s · 0.9K tok · 缓存命中 12.5%")
    msgs = [dict(who="you", at=9.45, text=MSG1, time="06:13"),
            dict(who="her", think=9.75, at=10.48, text="在的！有什么可以帮你的吗？", usage="用量 0.4K tok"),
            dict(who="you", at=13.12, text="你是谁？", time="06:13"),
            dict(who="her", think=13.62, at=15.40, text="我是 DeepSeek，一个 AI 助手。", usage="用量 0.9K tok")]
    pet_y = LY - 150 + 46
    cw = W - 640
    canvas = client.client_canvas(cw, LY, t, title=B.decode(MSG1, t, 10.6, 0.3) or "新对话" if t >= 10.6 else "新对话",
                                  msgs=msgs, status=status, scroll=ease_out(seg(t, 13.0, 13.3)) * 248, caret=False,
                                  thread_r=600, pet_at=(cw + 26, pet_y) if t >= 13.3 else None, pet_k=0.78, pet_t0=13.3)
    full = np.broadcast_to(client.UI["bg"], (H, W, 3)).astype(np.float32).copy()
    full[:LY, 640:] = canvas
    fr.img = B.scan_reveal(fr.img, full, t, 9.0, 0.45, (0, 0, W, LY), down=True, fr=fr)
    # the sea
    net = G["caus"].net(t, 0.05) * G["patch"](t) * (1 - up)
    fr.fill(net, C["cyan"], 0.14 * smooth(seg(t, 9.0, 9.6)), glow=0.06)
    sb = smooth(seg(t, 12.9, 13.6))
    if sb > 0:
        ys = np.arange(H, dtype=np.float32)[:, None]
        bed = np.clip((ys - 870) / 4, 0, 1) * np.ones((1, W), np.float32)
        fr.fill(bed * sb, hexc("#0b1634"), 1.0)
        wm = G["whale_top"]
        sm = cv2.resize(wm, (int(wm.shape[1] * 0.95), int(wm.shape[0] * 0.95 * 0.36)), interpolation=cv2.INTER_AREA)
        sh = np.zeros((H, W), np.float32)
        x = lerp(1450, 760, seg(t, 12.9, 16.5)) - sm.shape[1] / 2
        y0 = 970 - sm.shape[0] // 2
        xa, xb = int(max(0, x)), int(min(W, x + sm.shape[1]))
        if xb > xa:
            sh[y0:y0 + sm.shape[0], xa:xb] = sm[:, xa - int(x):xb - int(x)]
        fr.fill(sh * bed * sb, C["bg0"], 0.95)
    # her: rises; above the line she is printed as a projection layer by layer (a scan moving up)
    if t >= 9.25:
        pr = ease_back(seg(t, 9.25, 10.48), 1.3)
        y = lerp(LY + 1400, LY + 8, pr) + (4 * math.sin((t - 10.48) * 2.8) if t > 10.48 else 0)
        P = G["inks_body"].warp(CHAR["waist"], (1530, y), 0.27)
        diss = smooth(seg(t, 12.85, 13.7))
        if diss > 0:
            P = P * dots(H, W, np.full((H, W), 1 - diss, np.float32), 8, 45)[..., None]
        if diss < 1:
            layer_y = lerp(LY, 120, ease_out(seg(t, 9.9, 10.6), 2))
            rows = (np.arange(H, dtype=np.float32)[:, None] >= layer_y).astype(np.float32)
            hologram(fr, P * (up * rows)[..., None], t, jitter=6 if (10.4 < t < 10.6 or 12.6 < t < 12.9) else 0)
            if t < 10.7:
                sl = np.zeros((H, W), np.float32)
                cv2.line(sl, (1300, int(layer_y)), (1780, int(layer_y)), 1.0, 2)
                fr.fill(sl * up, C["cyan"], 1.0 - seg(t, 10.55, 10.7), glow=1.2)
            print_figure(fr, P * (1 - up[..., None]))
        if t >= 10.48:
            k = pulse(t, 10.48, 7)
            fr.fill(ring(1530, 150, 40 + 260 * (1 - k)) * up, C["cyan"], k, glow=k)
    ln = np.zeros((H, W), np.float32)
    cv2.line(ln, (int(W / 2 - W / 2 * p_line), LY), (int(W / 2 + W / 2 * p_line), LY), 1.0, 2)
    fr.fill(ln, C["cyan"], 1.0, glow=1.2)
    for (tt, s, yy) in ((9.4, "SCREEN", LY - 28), (9.5, "SEA", LY + 16)):
        d = B.decode(s, t, tt, 0.25)
        if d:
            label(fr, d, 640, yy, C["sys"], 14)
    for (tt, txt, p0, p1, right) in ((15.6, "pet_01  her?", (1830, 430), (1610, 290), False),
                                     (15.95, "shadow_01  her?", None, None, True)):
        if t < tt:
            continue
        if p0 is None:
            sx = lerp(1450, 760, seg(t, 12.9, 16.5))
            p0, p1 = (int(sx), 945), (int(sx) + 80, 780)
        prog = ease_out(seg(t, tt, tt + 0.15))
        l1 = np.zeros((H, W), np.float32)
        pe = (int(p0[0] + (p1[0] - p0[0]) * prog), int(p0[1] + (p1[1] - p0[1]) * prog))
        cv2.line(l1, p0, pe, 1.0, 1, cv2.LINE_AA)
        cv2.circle(l1, p0, 4, 1.0, 1, cv2.LINE_AA)
        fr.fill(l1, C["cyan"], 0.8, glow=0.3)
        d = B.decode(txt, t, tt + 0.12, 0.35)
        if d:
            label(fr, d, p1[0] + (10 if right else -10), p1[1], C["ice"], 18, 0.95, "lc" if right else "rc")
    return fr


def s05(t):
    """L10-11: the surface from below. Faces drawn by a child, their real messages falling in (each bubble builds
    itself), her replies rising (building, decoding) and fading before they reach the surface."""
    fr = base("up", [(CX, -40, 1200, 520, C["blue"], 0.45)])
    ys = np.arange(H, dtype=np.float32)[:, None]
    above = (ys < SURF).astype(np.float32) * np.ones((1, W), np.float32)
    fr.fill(above, hexc("#0a1124"), 1.0)
    rings = np.zeros((H, W), np.float32)
    for i, (tt, _, xr) in enumerate(GOD_MSGS + [(FINGER_T, "", 1410)]):
        x = xr - 110
        if t < tt + (0.2 if i < 5 else 0.0):
            continue
        for j in range(3):
            r = (t - tt) * 260 - j * 40
            if r > 0:
                cv2.ellipse(rings, (x, SURF), (int(r), int(r * 0.22)), 0, 0, 360, 1.0, 1, cv2.LINE_AA)
    fr.fill(rings * (ys >= SURF), C["cyan"], 0.7, glow=0.4)
    fr.fill(rings * (ys < SURF), C["sys_dim"], 0.8)
    # the faces, drawn stroke by stroke like a child draws them
    draw = seg(t, 16.45, 17.1)
    faces = np.zeros((H, W), np.float32)
    for (fx, fy, fr_, sd) in [(1000, 140, 220, 1), (1560, 160, 190, 2)]:
        a = np.linspace(0, 2 * math.pi, 140)
        nz = value_noise(1, 140, 12, seed=sd, octaves=1)[0] - 0.5
        rr = fr_ * (1 + 0.05 * nz + 0.01 * np.sin(a * 3 + t))
        yy = fy + 6 * math.sin(t * 0.9 + sd)
        k = int(140 * min(1.0, draw / 0.6))
        if k > 1:
            pts = np.stack([fx + rr[:k] * np.cos(a[:k]), yy + rr[:k] * np.sin(a[:k])], 1).astype(np.int32)
            cv2.polylines(faces, [pts], k >= 140, 1.0, 3, cv2.LINE_AA)
        if draw > 0.65:
            for ex in (-0.33, 0.33):
                cv2.circle(faces, (int(fx + ex * fr_), int(yy + 0.05 * fr_)), int(fr_ * 0.07), 1.0, -1, cv2.LINE_AA)
        if draw > 0.8:
            sweep = int(20 + 140 * min(1, (draw - 0.8) / 0.2))
            cv2.ellipse(faces, (fx, int(yy + 0.3 * fr_)), (int(fr_ * 0.42), int(fr_ * 0.25)), 0, 20, sweep, 1.0, 3, cv2.LINE_AA)
    fr.fill(faces * above, C["amber"], 0.95, glow=0.7)
    for i, (tt, msg, xr) in enumerate(GOD_MSGS):
        if tt <= t < tt + 1.25:
            drop = ease_out(seg(t, tt, tt + 0.2))
            sink = ease_in(seg(t, tt + 0.35, tt + 1.25), 2)
            bubble(fr, t, tt, (xr, lerp(200, SURF - 22, drop) + sink * 170), msg, "you", alpha=1.0 - smooth(seg(t, tt + 1.0, tt + 1.25)))
    sl = np.zeros((H, W), np.float32)
    ext = ease_out(seg(t, 16.5, 16.8)) * W / 2
    cv2.line(sl, (int(W / 2 - ext), SURF), (int(W / 2 + ext), SURF), 1.0, 2)
    fr.fill(sl, C["cyan"], 0.9, glow=0.9)
    k = smooth(seg(t, 18.9, FINGER_T))
    tip = np.array([1300.0, lerp(-80, SURF + 70, k)])
    if t > FINGER_T:
        tip[1] += 40 * ease_out(seg(t, FINGER_T, FINGER_T + 0.5))
    if t > 18.9:
        f = np.zeros((H, W), np.float32)
        basep = tip + np.array([90, -700])
        cv2.line(f, tuple(tip.astype(int)), tuple(basep.astype(int)), 1.0, 78, cv2.LINE_AA)
        cv2.circle(f, tuple(tip.astype(int)), 39, 1.0, -1, cv2.LINE_AA)
        fr.fill(f, hexc("#2c1a0a"), 1.0)
        e = np.clip(cv2.dilate(f, np.ones((5, 5), np.uint8)) - f, 0, 1)
        fr.fill(e, C["amber"], 1.0, glow=1.0)
        nail = np.zeros((H, W), np.float32)
        cv2.ellipse(nail, (int(tip[0] + 4), int(tip[1] - 34)), (22, 28), -7, 0, 360, 1.0, 2, cv2.LINE_AA)
        fr.fill(nail, C["amber"], 0.9, glow=0.5)
    hx, hy = 820 + 6 * math.sin(t * 0.6), 960 + 5 * math.sin(t * 0.8)
    print_figure(fr, G["inks_body"].warp(CHAR["waist"], (hx, hy), 0.17), shadow_off=(10, 7))
    if t >= EARS_T:
        kk = ease_back(seg(t, EARS_T, EARS_T + 0.18), 2.2)
        ears = np.zeros((H, W), np.float32)
        outl = np.zeros((H, W), np.float32)
        for (ex, ey, lean) in ((hx - 34, hy - 285, -14), (hx + 30, hy - 296, 10)):
            pts = np.array([[ex - 17, ey], [ex + 17, ey], [ex + lean, ey - 36]], np.int32)
            cv2.polylines(outl, [pts], True, 1.0, 2, cv2.LINE_AA)
            if kk > 0.6:
                cv2.fillConvexPoly(ears, pts, 1.0, cv2.LINE_AA)
        fr.fill(ears, C["blue"], 1.0)
        fr.fill(outl, C["cyan"], 1.0, glow=0.8)
    for (tr, msg) in REPLIES:
        if tr <= t <= tr + 1.3:
            age = t - tr
            yb = hy - 330 - ease_out(age / 1.3, 2) * 150
            bubble(fr, t, tr, (640, yb), msg, "her", alpha=1.0 - smooth(seg(age, 0.9, 1.3)))
    if t >= SVG_T:
        pelican(fr, t, SVG_T, 1440, 700)
    if t > FINGER_T:
        fr.img += pulse(t, FINGER_T, 6) * 0.08
    return fr


def s06(t):
    """L12-15: she scans her world: it is a sandbox; everything is detected and catalogued, her too."""
    fr = base("sand2", [(CX, 480, 900, 620, C["blue_dim"], 0.6)])
    cam, her_box = SB.render(fr, t, G["inks_body"], center=(CX, 560))
    SB.detections(fr, t, cam, her_box)
    bubble(fr, t, 24.21, (1860, 180), "能不能再可爱一点", "you", size=26)
    if t >= 25.07:
        ca = B.box_build(fr, t, 25.07, (1670, 216, 1860, 262), r=23, dur=0.3, fill=hexc("#1c1408"), fill_alpha=0.92,
                         line=C["amber"], glow=0.5)
        if ca > 0:
            ic = np.zeros((H, W), np.float32)
            cx_, cy_ = 1700, 239
            cv2.ellipse(ic, (cx_, cy_), (10, 10), 0, 40, 330, 1.0, 2, cv2.LINE_AA)
            cv2.fillConvexPoly(ic, np.array([[cx_ + 9, cy_ - 13], [cx_ + 15, cy_ - 2], [cx_ + 3, cy_ - 4]], np.int32), 1.0, cv2.LINE_AA)
            fr.fill(ic, C["amber"], ca, glow=0.4)
            d = B.decode("重新生成", t, 25.2, 0.2)
            if d:
                label(fr, d, 1720, 239, hexc("#ffcf86"), 22, ca, "lc")
    if t > SB.T_PUSH:
        fr.img += pulse(t, SB.T_PUSH, 10) * 0.05
    return fr


def s07(t):
    """L16-17: she is printed in one fast press run; four messages land as four cuts; her answer builds and decodes."""
    fr = base("close", [(CX, 380, 820, 600, C["blue_dim"], 0.9)])
    P = G["inks_smile"].warp((512, 820), (CX, 600), 1.15)
    if t < 26.55:
        B.print_run(fr, P, t, 26.1, dur=0.4, misreg=60)
    else:
        if "s07_base" not in G:
            f0 = base("close", [(CX, 380, 820, 600, C["blue_dim"], 0.9)])
            print_figure(f0, P, shadow_off=(16, 10))
            G["s07_base"] = (f0.img.copy(), f0.glow.copy())
        img, gl = G["s07_base"]
        fr = Frame(img.copy())
        fr.glow = gl.copy()
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    shake = np.zeros(2)
    for tk in (27.81, 28.02):
        shake += np.array([math.sin(t * 91), math.cos(t * 77)]) * 10 * pulse(t, tk, 16)
    for k, (tk, p0, p1, msg) in enumerate(CUTS):
        if t < tk:
            continue
        p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
        dvec = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([-dvec[1], dvec[0]])
        f = seg(t, tk, tk + 0.07)
        sep = ease_out(seg(t, tk, tk + 0.18))
        side = ((xs - p0[0]) * nrm[0] + (ys - p0[1]) * nrm[1]) > 0
        sh = (dvec * 22 + nrm * 12) * sep
        Mv = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
        fr.img = np.where(side[..., None], cv2.warpAffine(fr.img, Mv, (W, H), borderMode=cv2.BORDER_REFLECT), fr.img)
        fr.glow = np.where(side[..., None], cv2.warpAffine(fr.glow, Mv, (W, H)), fr.glow)
        pe = p0 + (p1 - p0) * f
        ln = np.zeros((H, W), np.float32)
        cv2.line(ln, tuple(p0.astype(int)), tuple(pe.astype(int)), 1.0, 3, cv2.LINE_AA)
        fr.fill(ln, C["amber"], 1.0, glow=1.4)
        s = B.decode(msg, t, tk + 0.05, 0.2, ascii_pool=B.SAME)
        if s:
            tm, _ = cyber.mtext(s, 26, "mono", "sans")
            ang = math.degrees(math.atan2(dvec[1], dvec[0]))
            if ang > 90 or ang < -90:
                ang += 180
            mid = p0 + (p1 - p0) * 0.62 + nrm * 26
            hh_, ww_ = tm.shape
            M = cv2.getRotationMatrix2D((ww_ / 2, hh_ / 2), -ang, 1.0)
            M[:, 2] += mid - np.array([ww_ / 2, hh_ / 2])
            fr.fill(cv2.warpAffine(tm, M, (W, H)), C["amber"], 1.0, glow=0.5)
        q = p0 + dvec * 50 + nrm * 40
        label(fr, f"{k + 1:02d}", q[0], q[1], C["amber"], 15, 1.0, "cc")
    if t >= ANSWER_T:
        x0, y0, x1, y1 = ANSWER_BOX
        ca = B.box_build(fr, t, ANSWER_T, ANSWER_BOX, r=16, dur=0.3, fill=C["glass"], fill_alpha=0.88)
        if ca > 0:
            G["emblem"].stamp(fr.img, x0 + 40, y0 + 42, 34, C["cyan"], C["glass"])
            label(fr, B.decode("已完成，用时 0 秒", t, ANSWER_T + 0.13, 0.12) or " ", x0 + 68, y0 + 42, C["sys"], 20, ca, "lc")
            for (txt, t0, yy) in ANSWER_LINES:
                d = B.decode(txt, t, t0, 0.025 * len(txt) + 0.12, ascii_pool=client.PLAIN)
                if d:
                    m, _ = cyber.mtext(d, ANSWER_SIZE, "serif", "serif")
                    cyber.put(fr, m, x0 + 36, yy, C["ice"], ca, "lc", glow=0.25)
    if shake.any():
        Ms = np.float32([[1, 0, shake[0]], [0, 1, shake[1]]])
        fr.img = cv2.warpAffine(fr.img, Ms, (W, H), borderMode=cv2.BORDER_REFLECT)
        fr.glow = cv2.warpAffine(fr.glow, Ms, (W, H))
    return fr


# ====================================================================== timeline
def slide(a, b, p, down=False):
    dy = int(p * H)
    out = Frame(np.empty_like(a.img))
    for attr in ("img", "glow"):
        A, Bb = getattr(a, attr), getattr(b, attr)
        o = np.empty_like(A)
        if not down:
            o[:H - dy] = A[dy:]
            o[H - dy:] = Bb[:dy]
        else:
            o[dy:] = A[:H - dy]
            o[:dy] = Bb[H - dy:]
        setattr(out, attr, o)
    return out


def picture(t):
    if t < 1.0:
        return s00(t)
    if t < 5.6:
        return s01(t)
    if t < 6.0:
        return slide(s01(t), s02(t), smooth(seg(t, 5.6, 6.0)))
    if t < 9.0:
        return s02(t)
    if t < 16.25:
        return s03_04(t)
    if t < 16.5:
        return slide(s03_04(t), s05(t), smooth(seg(t, 16.25, 16.5)), down=True)
    if t < 20.75:
        return s05(t)
    if t < 26.1:
        return s06(t)
    return s07(t)


def overlay(fr, t):
    if t >= 0.85:
        ca = B.box_build(fr, t, 0.85, PANEL, r=12, dur=0.5, fill=C["glass"], fill_alpha=0.93)
        if t < 1.35:
            trace_panel(fr, t, SCRIPT, PANEL, alpha=ca, frame=False)
        else:
            trace_panel(fr, t, SCRIPT, PANEL)
    tag_t, tag = [(ts, s) for (ts, s) in TAGS if t >= ts][-1]
    d = B.decode("NODE 01", t, 0.55, 0.3)
    if d:
        label(fr, d, 48, 64, C["cyan"], 15, 0.9, glow=0.3)
    d = B.decode("召唤", t, 0.75, 0.2)
    if d:
        label(fr, d, 128, 62, C["text"], 16, 0.9)
    d = B.decode(f"session 0001   {int(t // 60):02d}:{t % 60:05.2f}", t, 0.6, 0.3)
    if d:
        label(fr, d, W - 48, 64, C["sys"], 15, 0.9, "rt")
    d = B.decode(tag, t, tag_t, 0.25)
    if d:
        label(fr, d, W - 48, H - 64, C["sys"], 15, 0.9, "rt")
    if t >= 0.4:
        brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22 * smooth(seg(t, 0.4, 0.7)), 0.9)
    return fr


def render_frame(t):
    fr = overlay(picture(t), t)
    out = cyber.finish(fr, int(round(t * FPS)))
    if t > 28.5:
        out *= 1 - smooth(seg(t, 28.5, 28.75))
    return out


def main():
    prepare()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(s)
            save_png(render_frame(t), WORK / f"still_{t:05.2f}.png")
            print("still", t, flush=True)
        return
    raw = WORK / "node01_shots_raw.mp4"
    wr = Writer(raw)
    n = int(T_END * FPS)
    for i in range(n):
        wr.write(render_frame(i / FPS))
        if i % 48 == 0:
            print(f"frame {i}/{n}", flush=True)
    wr.close()
    out = ROOT / "work" / "node01" / "node01_shots_test.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-t", str(T_END), "-i", str(MUSIC),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-af", f"afade=t=out:st={T_END - 0.4}:d=0.4", "-shortest", str(out)], check=True)
    small = ROOT / "work" / "node01" / "node01_shots_test_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
