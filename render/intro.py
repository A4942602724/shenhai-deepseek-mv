"""The notice before the film: three silent seconds, in the film's own type, nothing flashing.

  python render/intro.py --stills 0.5 1.5 2.9   -> work/intro/still_*.png
  python render/intro.py --full                  -> work/intro/notice_raw.mp4 (72 frames; tools/assemble.py --lead)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build as B  # noqa: E402
import cyber  # noqa: E402
from cyber import C, Frame, label  # noqa: E402
from lib import FPS, H, ROOT, W, Writer, save_png, seg, smooth  # noqa: E402

WORK = ROOT / "work" / "intro"
WORK.mkdir(parents=True, exist_ok=True)
DUR = 3.0
HEAD = "【注意】"
LINES = ["视频内容纯属虚构，本片含闪烁画面与强光，光敏感者请留意。",
         "本视频内容包含轻微惊悚要素，如您在观看时感到不适，请立即退出本视频。"]


def render_frame(t):
    fr = Frame(cyber.ground("notice"))
    out = 1 - smooth(seg(t, DUR - 0.35, DUR))                 # into black, where the film begins
    a = smooth(seg(t, 0.0, 0.3)) * out
    cyber.brackets(fr, 24, 24, W - 24, H - 24, C["sys_dim"], 22, 0.9 * a)
    d = B.decode(HEAD, t, 0.10, 0.25)
    if d:
        label(fr, d, W / 2, H / 2 - 92, C["amber"], 40, out, "cc", glow=0.25 * out)
    for k, s_ in enumerate(LINES):
        d = B.decode(s_, t, 0.25 + 0.15 * k, 0.45)
        if d:
            label(fr, d, W / 2, H / 2 - 4 + 62 * k, C["text"], 32, out, "cc")
    img = cyber.finish(fr, int(t * FPS), vignette=0.4, scan=0.03, ca=0.0, grain=0.012)
    return np.clip(img * out + 0 * (1 - out), 0, 1)


def main():
    if "--stills" in sys.argv:
        for s in sys.argv[sys.argv.index("--stills") + 1:]:
            t = float(s)
            save_png(render_frame(t), WORK / f"still_{t:04.2f}.png")
            print("still", t)
        return
    if "--full" not in sys.argv:
        print("usage: --stills t ... | --full")
        return
    wr = Writer(WORK / "notice_raw.mp4")
    for i in range(int(round(DUR * FPS))):
        wr.write(render_frame(i / FPS))
    wr.close()
    print("done", WORK / "notice_raw.mp4")


if __name__ == "__main__":
    main()
