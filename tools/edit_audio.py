"""The film's edit of the song (2026-10-14, user's request): at 1:35.42, as her eyes close, the music is cut with a
quick fade-out; 3 s of silence; then straight into the high note at 1:41 (orig 101.42). Everything after shifts 3.00 s
earlier. Applied to the mix and to the Demucs stems.

  python tools/edit_audio.py     -> audio/music11_edit.wav, analysis/stems_edit/*.wav
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "audio" / "music11.mp3"            # the song itself (not in the repository: bring your own copy)
CUT, FADE, SILENCE, RESUME = 95.42, 0.25, 3.00, 101.42      # new time = orig - (RESUME - CUT - SILENCE) after the gap


def edit(src, dst):
    fc = (f"[0:a]atrim=0:{CUT},asetpts=PTS-STARTPTS,afade=t=out:st={CUT - FADE}:d={FADE}:curve=qsin[a];"
          f"anullsrc=r=48000:cl=stereo,atrim=0:{SILENCE}[s];"
          f"[0:a]atrim=start={RESUME},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.015[b];"
          f"[a]aresample=48000,aformat=channel_layouts=stereo[a2];[b]aresample=48000,aformat=channel_layouts=stereo[b2];"
          f"[a2][s][b2]concat=n=3:v=0:a=1[o]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-filter_complex", fc, "-map", "[o]", "-c:a", "pcm_s16le",
                    str(dst)], check=True)


def main():
    edit(SRC, ROOT / "audio" / "music11_edit.wav")
    for p in sorted((ROOT / "analysis" / "stems").glob("*.wav")):
        edit(p, ROOT / "analysis" / "stems_edit" / p.name)
    print("ok")


if __name__ == "__main__":
    main()
