"""Signal damage for whole frames: tearing, channel split, macroblock smear, vertical roll, dropped rows.
All ops take an (H, W, 3) float image and a numpy Generator, so a frame's damage is a pure function of its seed."""
import cv2
import numpy as np


def rng(t, salt=0, fps=24):
    return np.random.default_rng(int(round(t * fps)) * 7919 + salt)


def tear(img, r, n=6, max_shift=160, min_h=4, max_h=90):
    """Horizontal bands slid sideways (a torn signal)."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        y = int(r.uniform(0, h))
        bh = int(r.uniform(min_h, max_h))
        dx = int(r.normal(0, max_shift / 2.2))
        band = img[y:y + bh]
        out[y:y + bh] = np.roll(band, dx, axis=1)
    return out


def split(img, dx, dy=0):
    """Chromatic split: red one way, blue the other."""
    out = img.copy()
    M1 = np.float32([[1, 0, dx], [0, 1, dy]])
    M2 = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = img.shape[:2]
    out[..., 0] = cv2.warpAffine(img[..., 0], M1, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out[..., 2] = cv2.warpAffine(img[..., 2], M2, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out


def blocks(img, r, n=14, size=(32, 192), max_off=220):
    """Macroblock corruption: rectangles copied from somewhere nearby, some smeared flat."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        bw = int(r.choice([16, 32, 48, 64, 96, 128, 192]))
        bh = int(r.choice([16, 16, 32, 32, 48]))
        x, y = int(r.uniform(0, w - bw)), int(r.uniform(0, h - bh))
        sx = int(np.clip(x + r.normal(0, max_off / 2), 0, w - bw))
        sy = int(np.clip(y + r.normal(0, max_off / 6), 0, h - bh))
        if r.random() < 0.3:                                   # smeared: one source row stretched over the block
            out[y:y + bh, x:x + bw] = img[sy:sy + 1, sx:sx + bw]
        else:
            out[y:y + bh, x:x + bw] = img[sy:sy + bh, sx:sx + bw]
    return out


def roll(img, dy, bar=10):
    """Vertical hold slipping: the picture rolls, with a black sync bar where it wraps."""
    out = np.roll(img, int(dy), axis=0)
    y = int(dy) % img.shape[0]
    out[max(0, y - bar):y] = 0
    return out


def drop_rows(img, r, frac=0.05, dark=0.0):
    out = img.copy()
    h = img.shape[0]
    n = int(h * frac / 3)
    for _ in range(n):
        y = int(r.uniform(0, h))
        out[y:y + int(r.uniform(1, 5))] *= dark
    return out


def crush(img, levels=4):
    return np.round(np.clip(img, 0, 1) * (levels - 1)) / (levels - 1)


def damage(img, r, k=1.0):
    """A full hit of damage at strength k (0..1+)."""
    if k <= 0:
        return img
    out = img
    if r.random() < 0.6 * k:
        out = roll(out, r.uniform(-1, 1) * 180 * k)
    out = tear(out, r, n=int(3 + 9 * k), max_shift=60 + 220 * k)
    out = blocks(out, r, n=int(4 + 16 * k))
    out = split(out, (6 + 22 * k) * (1 if r.random() < 0.5 else -1), r.uniform(-2, 2))
    out = drop_rows(out, r, frac=0.03 + 0.08 * k)
    return out
