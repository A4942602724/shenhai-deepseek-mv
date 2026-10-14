"""Whale skeletons for node 05 (鲸墓), translated in code from public-domain lithographs: P.-J. van Beneden &
P. Gervais, Ostéographie des cétacés (1868-1880), plates in assets/ref/ (see SOURCES.md). Nothing is drawn freehand.

Each skeleton is a side view on a plate. Its bones are separated from the paper (ink = how much darker than the local
paper; enclosed outlines filled), then turned into a 3D point cloud: every column of the side view is read as a
cross-section of a barrel hanging from the spine, so a rib seen from the side becomes an arc going out and down on both
sides, the neural spines stay on the midline, the tail stays thin. Points carry their ink (the engraving's lines and
shading), so a scan of the cloud reads like the plate.

  python render/bones.py --debug   -> work/node05/bones_*.jpg (masks and clouds, for checking the crops)
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib import ROOT  # noqa: E402

REF = ROOT / "assets" / "ref"
CACHE = ROOT / "work" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
PREV_W = 1800.0                     # crop rectangles are given on a 1800-px-wide preview of each plate

PLATES = dict(balaena=REF / "VanBeneden_Gervais_PL01-02_Balaena_australis.jpg",
              megaptera=REF / "VanBeneden_Gervais_PL10-11_Megaptera.jpg",
              physeter=REF / "VanBeneden_Gervais_PL18_Physeter_Hyperoodon.jpg",
              phocaena=REF / "VanBeneden_Gervais_PL55_Phocaena.jpg")

# name: plate, crop (preview px), rectangles to blank out (other figures, labels), body length in metres
SPECS = dict(
    balaena_a=dict(plate="balaena", crop=(35, 120, 1752, 512), length=15.0,
                   out=[(28, 320, 245, 560), (425, 296, 512, 412), (592, 296, 702, 425), (772, 306, 882, 405),
                        (1318, 440, 1760, 712), (880, 92, 925, 124), (870, 288, 900, 312)]),
    balaena_b=dict(plate="balaena", crop=(20, 760, 1752, 1205), length=15.0,
                   out=[(60, 722, 245, 975), (292, 776, 372, 910), (560, 688, 698, 806), (640, 992, 772, 1188), (320, 1175, 790, 1210),
                        (1172, 656, 1248, 778), (812, 760, 836, 780), (795, 840, 820, 870)]),
    megaptera=dict(plate="megaptera", crop=(120, 110, 1712, 440), length=14.0,
                   out=[(118, 262, 322, 418), (115, 290, 942, 532), (1292, 392, 1702, 502), (893, 405, 1012, 668),
                        (855, 112, 880, 132)]),
    physeter=dict(plate="physeter", crop=(60, 760, 1760, 1100), length=16.0,
                  out=[(250, 672, 398, 885), (425, 640, 670, 866), (685, 680, 800, 826), (885, 660, 1105, 836),
                       (1358, 410, 1645, 872), (640, 1010, 700, 1030), (820, 1050, 960, 1078)]),
    hyperoodon=dict(plate="physeter", crop=(250, 150, 1615, 448), length=8.0,
                    out=[(800, 312, 1016, 416), (612, 368, 748, 478), (995, 420, 1147, 600), (895, 145, 915, 165),
                         (220, 385, 400, 450), (450, 380, 600, 450), (1395, 410, 1560, 450)]),
    phocaena=dict(plate="phocaena", crop=(305, 165, 1565, 478), length=1.7,
                  out=[(740, 280, 860, 336), (758, 328, 978, 528), (212, 380, 712, 585), (1283, 376, 1508, 708),
                       (850, 168, 870, 185), (1360, 168, 1380, 188), (1500, 385, 1570, 480),
                     (700, 370, 730, 415)]),
)
SKULL = dict(plate="balaena", crop=(262, 492, 548, 742), height=4.2,      # Pl. I-II fig. 9: the skull seen from behind
             out=[(390, 498, 412, 516), (390, 708, 412, 734)],          # the figure number and the scale
             hole=(432, 250, 45, 56))                                    # foramen magnum: centre, radii (900-px crop)

_PL = {}


def plate(key):
    if key not in _PL:
        im = cv2.imread(str(PLATES[key]), cv2.IMREAD_GRAYSCALE)
        _PL[key] = im
    return _PL[key]


def ink_map(gray):
    """How much darker than the paper around it, 0..1. The paper's own tone (foxing, the gutter) drops out."""
    g = gray.astype(np.float32)
    paper = cv2.dilate(g, np.ones((31, 31), np.uint8))
    paper = cv2.GaussianBlur(paper, (0, 0), 25)
    col = np.median(g, axis=0, keepdims=True)                 # the gutter's shadow runs down whole columns
    col = cv2.GaussianBlur(col, (0, 0), 3)
    paper = np.minimum(paper, np.maximum(col, 0.6 * paper))
    r = g / np.maximum(paper, 1)
    return np.clip((0.955 - r) / 0.55, 0, 1)


def crop(spec, work_w):
    """The figure, blanked of its neighbours, at work_w px wide: (ink, bone mask)."""
    G = plate(spec["plate"])
    s = G.shape[1] / PREV_W
    x0, y0, x1, y1 = [int(round(v * s)) for v in spec["crop"]]
    g = G[y0:y1, x0:x1].copy()
    for (a, b, c, d) in spec.get("out", []):
        a, b, c, d = int(a * s) - x0, int(b * s) - y0, int(c * s) - x0, int(d * s) - y0
        g[max(0, b):max(0, d), max(0, a):max(0, c)] = 255
    k = work_w / g.shape[1]
    g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ink = ink_map(g)
    m = (ink > 0.07).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # the figure is the big cluster of marks; drop what floats apart from it (labels, the edges of other figures)
    grp = cv2.dilate(m, np.ones((23, 23), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grp, 8)
    if n > 1:
        big = st[1:, cv2.CC_STAT_AREA].max()
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 0.04 * big
        m = m * keep[lab]
    frac = m.mean(0)                                          # what is left of the gutter: a thin full-height line
    ref = cv2.blur(frac[None, :].astype(np.float32), (31, 1))[0]
    gut = (frac > 0.3) & (frac > 3 * ref)
    gut = np.convolve(gut.astype(np.float32), np.ones(3), mode="same") > 0
    m[:, gut] = 0
    filled = m
    return ink * filled, filled.astype(np.float32)


def _smooth_env(v, valid, win):
    idx = np.arange(len(v))
    if valid.sum() < 2:
        return v
    v = np.interp(idx, idx[valid], v[valid])
    k = np.ones(win) / win
    return np.convolve(np.pad(v, win, mode="edge"), k, mode="same")[win:-win]


def cloud(name, work_w=1500, seed=0, density=0.55):
    """(n,3) points in metres (x head-wards, y lateral, z up, lowest bone at z=0, head at x=+L/2) and (n,) ink."""
    p = CACHE / f"bones_{name}_{work_w}.npz"
    if p.exists():
        d = np.load(p)
        return d["P"], d["I"]
    spec = SPECS[name]
    ink, mask = crop(spec, work_w)
    h, w = mask.shape
    cols = mask.any(0)
    top = np.where(cols, mask.argmax(0), 0).astype(np.float32)
    bot = np.where(cols, h - 1 - mask[::-1].argmax(0), 0).astype(np.float32)
    top = _smooth_env(top, cols, 9)
    bot = _smooth_env(bot, cols, 9)
    ys, xs = np.nonzero(mask > 0)
    I = ink[ys, xs]
    g = np.random.default_rng(seed)
    keep = g.random(len(xs)) < density * (0.18 + 0.82 * np.clip(I * 1.6, 0, 1))
    xs, ys, I = xs[keep], ys[keep], I[keep]
    d = ys - top[xs]
    D = np.maximum(bot[xs] - top[xs], 1.0)
    half = 0.86 * np.sqrt(np.clip(d * (D - d), 0, None))           # the barrel's half-width at that depth
    side = np.where(g.random(len(xs)) < 0.5, -1.0, 1.0)
    lat = side * half + g.normal(0, 0.9, len(xs))
    xs_ = xs.astype(np.float32) + g.uniform(-0.5, 0.5, len(xs))
    ys_ = ys.astype(np.float32) + g.uniform(-0.5, 0.5, len(xs))
    m_per_px = spec["length"] / w
    P = np.stack([(xs_ - w / 2) * m_per_px, lat * m_per_px, (ys_.max() - ys_) * m_per_px], 1).astype(np.float32)
    P[:, 2] -= P[:, 2].min()
    np.savez_compressed(p, P=P, I=I.astype(np.float32))
    return P, I.astype(np.float32)


def skull_slab(work_w=1300, seed=1, density=1.0):
    """Fig. 9 as an upright slab: (n,3) points (x across, y depth, z up, base at z=0), ink, and the hole (x, y, z, r).
    The face bulges a little towards the viewer so it is not a sheet of paper. The foramen magnum is left empty."""
    p = CACHE / f"bones_skull_{work_w}.npz"
    if p.exists():
        d = np.load(p)
        return d["P"], d["I"], d["hole"]
    ink, mask = crop(SKULL, work_w)
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0)
    I = ink[ys, xs]
    g = np.random.default_rng(seed)
    keep = g.random(len(xs)) < density * (0.25 + 0.75 * np.clip(I * 1.6, 0, 1))
    xs, ys, I = xs[keep], ys[keep], I[keep]
    k9 = work_w / 900.0
    hx, hy, rx, ry = [v * k9 for v in SKULL["hole"]]
    out = ((xs - hx) / (rx * 1.12)) ** 2 + ((ys - hy) / (ry * 1.12)) ** 2 > 1
    xs, ys, I = xs[out], ys[out], I[out]
    m = SKULL["height"] / h
    X = (xs - w / 2) * m
    Z = (h - ys) * m
    r = np.sqrt((X / (w * m / 2)) ** 2 + ((Z - h * m / 2) / (h * m / 2)) ** 2)
    Y = -0.9 * np.clip(1 - r * r, 0, 1) + g.normal(0, 0.03, len(xs))
    P = np.stack([X, Y, Z], 1).astype(np.float32)
    hX, hZ = (hx - w / 2) * m, (h - hy) * m
    hr = math.hypot(hX / (w * m / 2), (hZ - h * m / 2) / (h * m / 2))
    hole = np.array([hX, -0.9 * max(0.0, 1 - hr * hr) - 0.05, hZ, 0.5 * (rx + ry) * m], np.float32)
    np.savez_compressed(p, P=P, I=I.astype(np.float32), hole=hole)
    return P, I.astype(np.float32), hole


def debug():
    out = ROOT / "work" / "node05"
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in list(SPECS.items()) + [("skull", SKULL)]:
        ink, mask = crop(spec, 1500 if name != "skull" else 900)
        vis = np.dstack([mask * 90, ink * 255, ink * 255]).astype(np.uint8)
        cv2.imwrite(str(out / f"bones_{name}_mask.jpg"), vis)
        print(name, mask.shape, int(mask.sum()))
        if name == "skull":
            continue
        P, I = cloud(name)
        for view, (a, b) in dict(side=(0, 2), top=(0, 1), front=(1, 2)).items():
            span = P.max(0) - P.min(0)
            sc = 1400 / max(span[a], span[b])
            img = np.zeros((int(span[b] * sc) + 20, int(span[a] * sc) + 20), np.float32)
            u = ((P[:, a] - P[:, a].min()) * sc + 10).astype(int)
            v = (img.shape[0] - 10 - (P[:, b] - P[:, b].min()) * sc).astype(int)
            np.add.at(img, (v, u), 0.25 + I)
            cv2.imwrite(str(out / f"bones_{name}_{view}.jpg"), np.clip(img * 160, 0, 255).astype(np.uint8))
        print("  points", len(P))


if __name__ == "__main__":
    if "--debug" in sys.argv:
        debug()
