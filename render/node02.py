"""NODE 02 对话 · respond() — 0:31.45-1:07.76 「没见过的海」. Storyboard: docs/分镜_第二节.md.

  31.45-32.90  node 01's last frame held: the composer's caret slows and stops; she closes her eyes; the frame
               closes like an eyelid                                                    (n2_01 + node02_open.eyelid)
  32.90-41.95  black; a flash of her singing; the deep, where she sings at 52 Hz; the build's flashbacks
                                                                                         (node02_open.py)
  41.95-67.76  locked on her, small in a big sea, sinking from just under the surface into the dark; the cyber
               layer (tracking, failing uplink, sonar, scan, rising data); signal lost on the hit (descent.py)

  python render/node02.py --stills 32.8 36.5 ...   -> work/node02/still_*.png
  python render/node02.py --range A B              -> work/node02/node02_patch_A_B.mp4 (raw frames A..B-1)
  python render/node02.py --bridge                 -> node 01's frames 747-753 without its fade to black
  python render/node02.py --full                   -> drafts/node02_v3.mp4 (+ _mobile)
"""
import subprocess
import sys
from pathlib import Path

import math

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import client  # noqa: E402
import cursors as CU  # noqa: E402
import cyber  # noqa: E402
import glitch  # noqa: E402
import descent as DC  # noqa: E402
import node01 as N1  # noqa: E402
import node01_shots as S1  # noqa: E402
import node02_open as OP  # noqa: E402
from cyber import C, Frame, brackets, label, print_figure, trace_panel  # noqa: E402
from lib import FPS, H, ROOT, MUSIC, SRC, W, Writer, ease_in, ease_out, hexc, lerp, save_png, seg, smooth  # noqa: E402

VERSION = "v3"
T0, T_END = 31.45, 67.76
DROP = 41.95
WORK = ROOT / "work" / "node02"
WORK.mkdir(parents=True, exist_ok=True)
PANEL = S1.PANEL
CX = 1250
G = S1.G
N2 = {}

# the TRACE panel carries on from node 01 until the eyelid closes over it
SCRIPT2 = [(31.80, "  you.typing = false"), (32.32, "  // 好安静")]
SCRIPT = sorted([(ta, ln) for ta, ln in N1.SCRIPT if ta > 20] + SCRIPT2, key=lambda x: x[0])


def prepare():
    if N2:
        return
    S1.prepare()
    OP.prepare()
    DC.prepare()
    from plates import Inks
    N2["inks_closed"] = Inks(scale=1.0, t1=0.30, t2=0.70, faceless=False, path=S1.USER / "A1-1_闭眼特写.png", min_k=0.35)


# ================================================================ the bridge out of node 01
T_CARET_STOP, T_EYES = 32.30, OP.T_EYES


def n2_01(t):
    """Node 01's last frame holds: four cuts on her smile, her answer in its box. Under it the user's composer: the
    caret blinks slower and slower and stops half-lit. She closes her eyes."""
    if "s07_closed" not in N2:
        P = N2["inks_closed"].warp((512, 820), (CX, 600), 1.15)
        f0 = Frame(cyber.ground("close", [(CX, 380, 820, 600, C["blue_dim"], 0.9)]))
        print_figure(f0, P, shadow_off=(16, 10))
        N2["s07_closed"] = (f0.img.copy(), f0.glow.copy())
        S1.s07(31.40)                                   # make sure the open-eyed base is cached
        N2["s07_open"] = G["s07_base"]
    G["s07_base"] = N2["s07_closed"] if t >= T_EYES else N2["s07_open"]
    fr = S1.s07(31.40)                                  # the settled end state of node 01, held
    G["s07_base"] = N2["s07_open"]
    x0, y0, x1, y1 = CX - 330, 994, CX + 330, 1046
    a = B.box_build(fr, t, 31.42, (x0, y0, x1, y1), r=20, dur=0.32, fill=C["glass"], fill_alpha=0.88)
    d = B.decode("发消息或创建任务", t, 31.62, 0.2, ascii_pool=client.PLAIN)
    if d:
        label(fr, d, x0 + 40, (y0 + y1) / 2, C["sys_dim"], 18, a, "lc")
    period = lerp(0.5, 1.1, smooth(seg(t, 31.40, T_CARET_STOP)))
    on = ((t - 31.40) % period) < period * 0.55
    lit = (1.0 if on else 0.0) if t < T_CARET_STOP else 0.45
    car = np.zeros((H, W), np.float32)
    cv2.rectangle(car, (x0 + 24, y0 + 13), (x0 + 27, y1 - 13), 1.0, -1)
    fr.fill(car, C["amber"], lit * a, glow=0.6 * lit * a)
    return fr


def overlay(fr, t):
    trace_panel(fr, t, SCRIPT, PANEL)
    for (s_, col, x, y, size) in (("NODE 01", C["cyan"], 48, 64, 15), ("召唤", C["text"], 128, 62, 16)):
        label(fr, s_, x, y, col, size, 0.9, "lt")
    label(fr, f"session 0001   {int(t // 60):02d}:{t % 60:05.2f}", W - 48, 64, C["sys"], 15, 0.9, "rt")
    label(fr, "L16–17", W - 48, H - 64, C["sys"], 15, 0.9, "rt")
    brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22, 0.9)
    return fr


# ================================================================ 41.64-41.95: the camera whips up out of the deep
T_WHIP = 41.64


def whip(t):
    """From her singing in the deep the camera tips straight up, faster and faster, through the lit surface, and lands
    on the sky over the sea as the chorus drops."""
    if "sky" not in N2:
        N2["sky"] = DC.render(DROP)
    a, b = OP.frame(t), N2["sky"]
    band_h = 260
    ys = np.linspace(0, 1, band_h, dtype=np.float32)[:, None, None]
    band = (C["ice"] * (1 - ys) + hexc("#2a86d6") * ys) * np.ones((1, W, 1), np.float32)
    canvas = np.concatenate([b, band, a], axis=0)
    e = ease_in(seg(t, T_WHIP, DROP), 2.4)
    y0 = int(round((H + band_h) * (1 - e)))
    view = canvas[y0:y0 + H]
    L = int(3 + 150 * e ** 0.8) | 1
    view = cv2.blur(view, (1, L))
    view = view + C["ice"] * 0.25 * math.sin(math.pi * e)
    return np.clip(view, 0, 1)


# ================================================================ 63.10-67.76: the other side of the screen
T_CUT = DC.HIT + 0.17                                  # a few torn frames of "signal lost", then the cut
T_CLICK = DC.PULSES[12] if hasattr(DC, "PULSES") else DC.HIT + 0.03 + 12 * DC.BEAT
PULSES = [DC.HIT + 0.03 + k * DC.BEAT for k in range(15)]
THREAD = [dict(who="you", at=-10, text="你好。你在吗？", time="06:13"),
          dict(who="her", think=-9.6, at=-9, text="在的！有什么可以帮你的吗？", usage="用量 0.4K tok", instant=True),
          dict(who="you", at=-8, text="你是谁？", time="06:13"),
          dict(who="her", think=-7.6, at=-7, text="我是 DeepSeek，一个 AI 助手。", usage="用量 0.9K tok", instant=True),
          dict(who="you", at=-6, text="反正你也记不住", time="06:14"),
          dict(who="her", think=-5.6, at=-5, text="我没办法保证。这次对话结束后，我不会记得。但现在，我在。", usage="用量 3.4K tok",
               instant=True)]


# what the user types and takes back: (time, text in the box after this keystroke). IME commits come in pairs.
TYPED = "我想让你记住所有"
EDITS = ([(63.40, "我想"), (63.63, "我想让你"), (63.82, "我想让你记住"), (64.05, TYPED)] +
         [(tk, TYPED[:len(TYPED) - k - 1]) for k, tk in enumerate((65.50, 65.58, 65.65, 65.72, 65.80, 65.87, 65.95, 66.10))])
T_MOVE = 66.12                                          # the hand goes back to the mouse


def typed_at(t):
    s_, last = "", -9.0
    for (te, txt) in EDITS:
        if t >= te:
            s_, last = txt, te
    return s_, last


def other_side(t):
    """The user's screen: the conversation where it stopped. In the box they type 我想让你记住所有, sit with it while
    the caret blinks on the bass, take it back one character at a time, then go up to the close button and close."""
    i = int(round(t * FPS))
    if t < T_CUT:
        out = DC.render(t)
        return np.clip(glitch.damage(out, glitch.rng(t, 21), 0.75), 0, 1)
    T_CLICK_ = PULSES[12]
    typed, last = typed_at(t)
    if t - last < 0.45:
        on = True                                       # the caret holds steady while keys are going
    else:
        on = any(0 <= t - p < 0.17 for p in PULSES)     # idle: it blinks on the bass
    canvas = client.client_canvas(W, H, 0.1 if on else 0.5, title="你好。你在吗？", msgs=THREAD, caret=True, typed=typed,
                                  status="7 轮 12 步 · 0 tok/s · 3.4K tok · 缓存命中 0.0%", thread_r=1700)
    fr = Frame(canvas)
    # the mouse: resting by the box while they type; afterwards up to the close button, a pause, a click
    t_in = T_CUT + 0.1
    cx, cy = W - 60, 29
    p0, p1 = (1330.0, 985.0), (cx + 2.0, cy + 2.0)
    u = smooth(seg(t, T_MOVE, T_CLICK_ - 0.28))
    x = lerp(p0[0], p1[0], u) + 30 * math.sin(u * math.pi)
    y = lerp(p0[1], p1[1], u ** 0.85)
    if T_CLICK_ - 0.28 <= t < T_CLICK_:                     # it waits over the button
        x += 4 * math.sin(t * 9.0)
        y += 2 * math.sin(t * 6.0)
    press = 1.0 if T_CLICK_ <= t < T_CLICK_ + 0.08 else 0.0
    CU.paste(fr, CU.patch("arrow", 4, t, t_in, seed=3, press=press), x, y)
    CU.click_ring(fr, x, y, t, T_CLICK_)
    out = cyber.finish(fr, i, vignette=0.35, scan=0.03)
    # the window closes like an eye
    if t >= T_CLICK_ + 0.05:
        k = ease_in(seg(t, T_CLICK_ + 0.05, T_CLICK_ + 0.25), 2.0)
        rows = np.arange(H, dtype=np.float32)[:, None]
        top, bot = (H / 2) * k, H - (H / 2) * k
        m = np.clip((rows - top) / 2, 0, 1) * np.clip((bot - rows) / 2, 0, 1)
        out = out * m[..., None]
        la = (1 - smooth(seg(t, T_CLICK_ + 0.25, T_CLICK_ + 0.55))) * (k > 0.97)
        if la > 0:
            xs = np.arange(W, dtype=np.float32)[None, :]
            line = np.exp(-((rows - H / 2) / 1.5) ** 2) * np.exp(-((xs - W / 2) / 600) ** 2)
            out = out + (line * la)[..., None] * C["ice"]
    return np.clip(out, 0, 1)


def render_frame(t):
    if t < OP.T_BLACK:
        out = cyber.finish(overlay(n2_01(t), t), int(round(t * FPS)))
        return OP.eyelid(out, t)
    if t < T_WHIP:
        return OP.frame(t)
    if t < DROP:
        return whip(t)
    if t >= DC.HIT:
        return other_side(t)
    return DC.render(t)


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
    if "--bridge" in sys.argv:                            # node 01's frames 747-753 without the fade to black
        N1.FADE_OUT = False
        wr = Writer(WORK / "node01_tail_nofade.mp4")
        for i in range(747, 754):
            wr.write(N1.render_frame(i / FPS))
        wr.close()
        print("bridge done")
        return
    raw = WORK / f"node02_{VERSION}_raw.mp4"
    n0, n1 = int(T0 * FPS), int(round(T_END * FPS))       # n0 = 754: the frame right after node 01's last one
    if "--range" in sys.argv:
        k = sys.argv.index("--range")
        n0, n1 = int(sys.argv[k + 1]), int(sys.argv[k + 2])
        raw = WORK / f"node02_patch_{n0}_{n1}.mp4"
    wr = Writer(raw)
    for i in range(n0, n1):
        wr.write(render_frame(i / FPS))
        if (i - n0) % 48 == 0:
            print(f"frame {i - n0}/{n1 - n0}", flush=True)
    wr.close()
    if "--range" in sys.argv:
        print("patch done", raw)
        return
    out = ROOT / "drafts" / f"node02_{VERSION}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ss", str(n0 / FPS), "-t", str((n1 - n0) / FPS),
                    "-i", str(MUSIC), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", str(out)], check=True)
    small = ROOT / "drafts" / f"node02_{VERSION}_mobile.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-preset", "slow", "-b:v", "6500k",
                    "-maxrate", "8000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(small)], check=True)
    print("done", out)


if __name__ == "__main__":
    main()
