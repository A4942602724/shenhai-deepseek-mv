"""Join rendered nodes into one preview with the music laid continuously underneath.

  python tools/assemble.py OUT.mp4 [--lead notice.mp4] node01.mp4 node02_raw.mp4 ...

Inputs are joined by video stream only (each one's own audio is ignored); the soundtrack is the film's edit of music11 (audio/music11_edit.wav) from 0:00
for the joined length. Also writes OUT_mobile.mp4 sized to stay under ~28 MB (the phone upload limit).
"""
import json
import subprocess
import sys
from pathlib import Path

MUSIC = Path(__file__).resolve().parents[1] / "audio" / "music11_edit.wav"   # the film's edit (tools/edit_audio.py)


def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def main():
    argv = sys.argv[1:]
    lead = None
    if "--lead" in argv:                       # a silent card before the film: the music starts after it
        k = argv.index("--lead")
        lead = Path(argv[k + 1])
        del argv[k:k + 2]
    out, parts = Path(argv[0]), [Path(p) for p in argv[1:]]
    delay_ms = int(round(duration(lead) * 1000)) if lead else 0
    if lead:
        parts = [lead] + parts
    n = len(parts)
    args = ["ffmpeg", "-y", "-v", "error"]
    for p in parts:
        args += ["-i", str(p)]
    args += ["-i", str(MUSIC)]
    delay = f"adelay={delay_ms}:all=1," if delay_ms else ""
    chain = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v];[{n}:a]{delay}apad[a]"   # silence after the song
    args += ["-filter_complex", chain, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", str(out)]
    subprocess.run(args, check=True)
    total = duration(out)
    # the phone copy: 720p, two-pass, under ~27.5 MiB (1080p at this bitrate turns the point clouds to blocks)
    vbit = int((27.5 * 8 * 1024 * 1024 / total - 128_000) * 0.97)
    small = out.with_name(out.stem + "_mobile.mp4")
    log = out.with_name(out.stem + "_2pass")
    common = ["-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-b:v", str(vbit), "-passlogfile", str(log)]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out)] + common + ["-pass", "1", "-an", "-f", "mp4", "NUL"], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out)] + common +
                   ["-pass", "2", "-maxrate", str(int(vbit * 1.5)), "-bufsize", str(vbit * 3), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", str(small)], check=True)
    for f in out.parent.glob(log.name + "*"):
        f.unlink()
    print(f"{out}  {total:.2f}s   mobile video bitrate {vbit / 1e6:.2f} Mb/s")


if __name__ == "__main__":
    main()
