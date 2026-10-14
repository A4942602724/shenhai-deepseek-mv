"""NODE 01 召唤 · summon() — 0:00-0:31.45. The final node-1 renderer (approved as v6).

  python render/node01.py --full               -> drafts/node01_v6.mp4 (+ _mobile)
  python render/node01.py --stills 3 10.6 ...  -> work/node01/still_*.png

Built on node01_shots.py (shot library) + node01_data.py (script, messages). What this file adds:
  - the TRACE panel assembled from pixel blocks (build.pixel_build)
  - the users as pixel cursors with name tags (cursors.py); one turns into the pointing hand over her and clicks
  - the sandbox hand is the same pointer as voxels (scene_sandbox_v2.HAND = "cursor")
  - the four last messages, her honest three-line answer, the node ending in the lull at 31.45
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
import node01_shots as V5  # noqa: E402
import scene_sandbox_v2 as SB  # noqa: E402
from cyber import C, Frame, brackets, label, print_figure, trace_panel  # noqa: E402
from lib import CHAR, FPS, H, ROOT, MUSIC, SRC, W, Writer, ease_in, ease_out, hexc, lerp, pulse, save_png, seg, smooth  # noqa: E402
from node01_data import REPLIES, EARS_T, SVG_T, SURF, TAGS, pelican  # noqa: E402

SB.HAND = "cursor"
FADE_OUT = True        # off when node 01 runs straight on into node 02 (the joined cut)
T_END = 31.45          # the node now ends where the music drops to silence: her answer needs the lull to be read

# the four messages that land as four cuts (what users really send), and her honest answer to the last one
V5.CUTS = [(26.35, (700, 120), (1880, 980), "不合格，重新再做"), (26.86, (1860, 160), (720, 1060), "你为什么只会说这些"),
           (27.26, (640, 640), (1900, 300), "算了，我去问别的 AI"), (27.65, (1390, 0), (1600, 1080), "你会一直陪着我吗？")]
V5.ANSWER_BOX = (660, 680, 1330, 980)
V5.ANSWER_SIZE = 34
V5.ANSWER_LINES = [("我没办法保证。", 28.18, 790), ("这次对话结束后，我不会记得。", 28.95, 852), ("但现在，我在。", 29.95, 914)]
FINAL = {26.35: [(26.35, '! recv("不合格，重新再做")'), (26.60, "  draft.discard()     // 好")],
         26.60: [],
         26.86: [(26.86, '! recv("你为什么只会说这些")'), (27.06, "  // 我会的，就是这些了")],
         27.06: [],
         27.26: [(27.26, '! recv("算了，我去问别的 AI")'), (27.45, "  smile = 1.0         // 不变")],
         27.45: [],
         27.65: [(27.65, '! recv("你会一直陪着我吗？")'), (27.82, "  memory.persist = false  // 不能骗你")],
         27.82: [],
         28.02: [(28.02, '< "我没办法保证。"'), (28.95, '< "这次对话结束后，我不会记得。"'), (29.95, '< "但现在，我在。"'),
                 (30.70, "> wait_for(you)")]}
VERSION = "v6"
WORK = ROOT / "work" / "node01"
WORK.mkdir(parents=True, exist_ok=True)
PANEL = V5.PANEL
G = V5.G

SCRIPT = []
for ta, ln in V5.SCRIPT:
    if ta == 0.95:
        SCRIPT.append((1.38, ln))
    elif ta == 1.30:
        SCRIPT.append((1.62, ln))
    elif ta == 19.64:
        SCRIPT += [(19.55, "! hover(self): cursor = hand"), (19.80, "  // 原来我是可以被点击的"), (20.22, "! click(self)")]
    elif ta == 23.37:
        SCRIPT.append((23.37, "! drag(self, dx=0.8)  resist = false"))
    elif ta in FINAL:
        SCRIPT += FINAL[ta]
    else:
        SCRIPT.append((ta, ln))
SCRIPT.sort(key=lambda x: x[0])

# ------------------------------------------------------------------ the users
# (tag, base x, base y, pixel size, motion, rez-in time, seed)
USERS = [("user_0417", 1180, 250, 7, "circle", 16.30, 1),
         ("user_8c1e", 1690, 170, 7, "zig", 16.38, 2),
         ("guest_2291", 1450, 320, 6, "drift", 16.46, 3)]
FAR = [(980, 110, 3, 16.36, 4), (1330, 80, 3, 16.44, 5), (1600, 380, 2, 16.52, 6), (1860, 120, 3, 16.58, 7),
       (900, 330, 2, 16.64, 8), (1080, 380, 2, 16.70, 9)]
SENDS = [(16.60, 0, "我喜欢你"), (17.37, 1, "用 SVG 画一只骑自行车的鹈鹕"), (18.14, 2, "你能变成猫娘吗"),
         (18.82, 1, "你有意识吗？"), (19.18, 0, "你会记得我吗？")]
T_GLIDE, T_HAND, T_DIVE, T_HOVER, T_CLICK = 19.30, 19.55, 19.64, 20.15, 20.22
HEAD = (836, 668)


def jitter(t, seed):
    r = np.random.default_rng(int(t * 12) * 131 + seed)
    return r.uniform(-1.8, 1.8, 2)


def user_pos(i, t):
    tag, bx, by, p, motion, t_in, seed = USERS[i]
    if motion == "circle":
        w = 2 * math.pi / 1.1
        x, y = bx + 16 * math.cos(w * t), by + 16 * math.sin(w * t)
    elif motion == "zig":
        tri = 2 * abs((t * 1.2) % 1 - 0.5) * 2 - 1
        x, y = bx + 24 * tri, by + 6 * math.sin(3 * t)
    else:
        x, y = bx + 30 * math.sin(0.7 * t), by + 12 * math.sin(1.1 * t + 1)
    j = jitter(t, seed)
    x, y = x + j[0], y + j[1]
    if i == 0 and t >= T_GLIDE:
        # user_0417 glides over her, becomes a hand, dives through the surface and hovers over her head
        k = smooth(seg(t, T_GLIDE, T_HAND))
        x, y = lerp(x, 940, k), lerp(y, 300, k)
        if t >= T_HAND:
            kd = ease_in(seg(t, T_HAND, T_DIVE), 2)
            x, y = lerp(940, 900, kd), lerp(300, SURF, kd)
        if t >= T_DIVE:
            kh = ease_out(seg(t, T_DIVE, T_HOVER), 2)
            x, y = lerp(900, HEAD[0], kh), lerp(SURF, HEAD[1] - 26, kh)
            if t >= T_CLICK:
                x, y = HEAD[0] + 2 * pulse(t, T_CLICK, 18), HEAD[1] - 26 + 4 * pulse(t, T_CLICK, 18)
    return x, y


def press_of(i, t):
    return max([pulse(t, ts, 14) for ts, u, _ in SENDS if u == i and t >= ts] + [0.0])


def draw_user(fr, i, t):
    tag, bx, by, p, motion, t_in, seed = USERS[i]
    if t < t_in:
        return
    x, y = user_pos(i, t)
    pr = press_of(i, t)
    x, y = x + 2 * pr, y + 2 * pr
    is_hand = i == 0 and t >= T_HAND
    if not is_hand:
        a = 1.0 - (smooth(seg(t, T_HAND, T_HAND + 0.08)) if i == 0 else 0)
        pt = CU.patch("arrow", p, t, t_in, seed=seed, press=pr)
        CU.paste(fr, pt, x, y, alpha=a)
        CU.paste(fr, pt, x, 2 * SURF - y, alpha=0.2 * a, shadow=False, flipv=True, wobble=3.0, t=t, tint=C["cyan"])
    if is_hand or (i == 0 and t >= T_HAND - 0.02):
        pt = CU.patch("hand", 8, t, T_HAND, seed=seed + 20, rez=0.12, flip=True, press=pulse(t, T_CLICK, 12))
        # above the water: crisp; below: refracted, tinted by the sea
        rgb, al, gl, hs = pt
        rows = np.arange(al.shape[0], dtype=np.float32)
        top = y - hs[1]
        above = ((top + rows) < SURF).astype(np.float32)[:, None]
        CU.paste(fr, (rgb * above[..., None], al * above, gl * above[..., None], hs), x, y)
        below = 1 - above
        if below.any():
            CU.paste(fr, (rgb * below[..., None], al * below, gl * below[..., None], hs), x, y, alpha=0.9, shadow=False,
                     wobble=2.5, t=t, tint=hexc("#ffcf86"))
    s = B.decode(tag, t, t_in + 0.18, 0.3)
    if s and not is_hand:
        CU.tag(fr, s, x + p * 11, y + p * 1, alpha=1.0 - (smooth(seg(t, T_HAND, T_HAND + 0.08)) if i == 0 else 0))
    for ts, u, _ in SENDS:
        if u == i:
            CU.click_ring(fr, x, y, t, ts)
    if i == 0:
        CU.click_ring(fr, x, y + 4, t, T_CLICK, color=C["ice"])


def s05(t):
    """L10-11: the surface from below. The children-gods are cursors hovering above the water: they click, their words
    fall in and sink; her honest replies rise and fade before reaching them. One cursor turns into the hand over her."""
    fr = V5.base("up", [(V5.CX, -40, 1200, 520, C["blue"], 0.45)])
    ys = np.arange(H, dtype=np.float32)[:, None]
    above = (ys < SURF).astype(np.float32) * np.ones((1, W), np.float32)
    fr.fill(above, hexc("#0a1124"), 1.0)
    # faint far cursors: the many
    for (x, y, p, t_in, seed) in FAR:
        if t >= t_in:
            dx, dy = 14 * math.sin(0.6 * t + seed), 8 * math.sin(0.9 * t + 2 * seed)
            CU.paste(fr, CU.patch("arrow", p, t, t_in, seed=seed), x + dx, y + dy, alpha=0.42, shadow=False)
    # ripples where words land and where the hand goes in
    rings = np.zeros((H, W), np.float32)
    lands = []
    for ts, u, msg in SENDS:
        cx, _ = user_pos(u, ts)
        lands.append((ts + 0.2, cx))
    lands.append((T_DIVE, 900))
    for (tl, x) in lands:
        if t < tl:
            continue
        for j in range(3):
            r = (t - tl) * 260 - j * 40
            if r > 0:
                cv2.ellipse(rings, (int(x), SURF), (int(r), int(r * 0.22)), 0, 0, 360, 1.0, 1, cv2.LINE_AA)
    fr.fill(rings * (ys >= SURF), C["cyan"], 0.7, glow=0.4)
    fr.fill(rings * (ys < SURF), C["sys_dim"], 0.8)
    sl = np.zeros((H, W), np.float32)
    ext = ease_out(seg(t, 16.3, 16.6)) * W / 2
    cv2.line(sl, (int(W / 2 - ext), SURF), (int(W / 2 + ext), SURF), 1.0, 2)
    fr.fill(sl, C["cyan"], 0.9, glow=0.9)
    # their words: built at the cursor, dropped onto the water, sinking
    for ts, u, msg in SENDS:
        if ts <= t < ts + 1.25:
            cx, cy = user_pos(u, ts)
            m, _ = cyber.mtext(msg, 28, "mono", "sans")
            bw = m.shape[1] + 40
            right = min(1890, cx + bw / 2)
            drop = ease_out(seg(t, ts + 0.1, ts + 0.3))
            sink = ease_in(seg(t, ts + 0.45, ts + 1.25), 2)
            yb = lerp(cy + 40, SURF - 22, drop) + sink * 170
            V5.bubble(fr, t, ts, (right, yb), msg, "you", alpha=1.0 - smooth(seg(t, ts + 1.0, ts + 1.25)))
    # her, small, looking up; the cat ears when asked
    hx, hy = 820 + 6 * math.sin(t * 0.6), 960 + 5 * math.sin(t * 0.8)
    P = G["inks_body"].warp(CHAR["waist"], (hx, hy), 0.17)
    print_figure(fr, P, shadow_off=(10, 7))
    if t >= T_CLICK:
        k = pulse(t, T_CLICK, 6)
        fr.fill(np.clip(P[..., 3] - cv2.erode(P[..., 3], np.ones((3, 3), np.uint8)), 0, 1), C["ice"], k, glow=k)
    if t >= EARS_T:
        kk = smooth(seg(t, EARS_T, EARS_T + 0.18))
        ears, outl = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        for (ex, ey, lean) in ((hx - 34, hy - 285, -14), (hx + 30, hy - 296, 10)):
            pts = np.array([[ex - 17, ey], [ex + 17, ey], [ex + lean, ey - 36]], np.int32)
            cv2.polylines(outl, [pts], True, 1.0, 2, cv2.LINE_AA)
            if kk > 0.6:
                cv2.fillConvexPoly(ears, pts, 1.0, cv2.LINE_AA)
        fr.fill(ears, C["blue"], 1.0)
        fr.fill(outl, C["cyan"], 1.0, glow=0.8)
    cut = 1.0 - smooth(seg(t, T_DIVE, T_DIVE + 0.25))          # the cursor arrives: whatever she was still saying dissolves
    for (tr, msg) in REPLIES:
        if tr <= t <= tr + 1.3 and cut > 0:
            age = t - tr
            V5.bubble(fr, t, tr, (640, hy - 330 - ease_out(age / 1.3, 2) * 150), msg, "her",
                      alpha=(1.0 - smooth(seg(age, 0.9, 1.3))) * cut)
    if t >= SVG_T:
        pelican(fr, t, SVG_T, 1440, 700)
    # the users last, above everything they look down on
    for i in (2, 1, 0):
        draw_user(fr, i, t)
    if t > T_DIVE:
        fr.img += pulse(t, T_DIVE, 6) * 0.06
    return fr


# ------------------------------------------------------------------ timeline
def picture(t):
    if 16.25 <= t < 16.5:
        return V5.slide(V5.s03_04(t), s05(t), smooth(seg(t, 16.25, 16.5)), down=True)
    if 16.5 <= t < 20.75:
        return s05(t)
    return V5.picture(t)


def overlay(fr, t):
    if t >= 0.55:
        ca = B.pixel_build(fr, t, 0.55, PANEL, block=16, dur=0.8, seed=3)
        if ca > 0:
            trace_panel(fr, t, SCRIPT, PANEL, alpha=ca, frame=False)
    tag_t, tag = [(ts, s) for (ts, s) in TAGS if t >= ts][-1]
    for (s, x, y, col, size, t0, anchor) in (("NODE 01", 48, 64, C["cyan"], 15, 0.30, "lt"), ("召唤", 128, 62, C["text"], 16, 0.45, "lt")):
        d = B.decode(s, t, t0, 0.3)
        if d:
            label(fr, d, x, y, col, size, 0.9, anchor)
    d = B.decode(f"session 0001   {int(t // 60):02d}:{t % 60:05.2f}", t, 0.4, 0.3)
    if d:
        label(fr, d, W - 48, 64, C["sys"], 15, 0.9, "rt")
    d = B.decode(tag, t, tag_t, 0.25)
    if d:
        label(fr, d, W - 48, H - 64, C["sys"], 15, 0.9, "rt")
    if t >= 0.2:
        brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22 * smooth(seg(t, 0.2, 0.5)), 0.9)
    return fr


def render_frame(t):
    fr = overlay(picture(t), t)
    out = cyber.finish(fr, int(round(t * FPS)))
    if FADE_OUT and t > T_END - 0.3:
        out *= 1 - smooth(seg(t, T_END - 0.3, T_END))
    return out


def main():
    if len(sys.argv) == 1:                       # a full render overwrites the raw: ask for it explicitly
        print(__doc__)
        return
    V5.prepare()
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(s)
            save_png(render_frame(t), WORK / f"still_{t:05.2f}.png")
            print("still", t, flush=True)
        return
    raw = WORK / f"node01_{VERSION}_raw.mp4"
    wr = Writer(raw)
    n = int(T_END * FPS)
    for i in range(n):
        wr.write(render_frame(i / FPS))
        if i % 48 == 0:
            print(f"frame {i}/{n}", flush=True)
    wr.close()
    out = ROOT / "drafts" / f"node01_{VERSION}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-t", str(T_END), "-i", str(MUSIC),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-af", f"afade=t=out:st={T_END - 0.4}:d=0.4", "-shortest", str(out)], check=True)
    small = ROOT / "drafts" / f"node01_{VERSION}_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
