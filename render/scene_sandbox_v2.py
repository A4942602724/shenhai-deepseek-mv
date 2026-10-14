"""S06 v2: the sandbox as a LiDAR-like scan. Point clouds (toys, the hand), scan-line sand, a sweep that builds the
world, object-detection boxes that catalogue everything in it (her included, with low confidence).
Software 3D only (sandbox3d.Cam for projection)."""
import math

import cv2
import numpy as np

from build import decode
from cyber import C, label, print_figure
from lib import CHAR, H, W, ease_back, ease_in, ease_out, hexc, lerp, seg, smooth
from plates import dots
import sandbox3d as S3

HER0 = np.array([-0.35, -0.20])
HER1 = np.array([0.42, -0.36])
HER_H = 0.62
T_SCAN0, T_SCAN1 = 20.85, 21.60
T_PUSH, T_PUSHED, T_REGEN = 23.37, 24.90, 25.07
HAND = "finger"            # "finger": a child's hand (v5); "cursor": the users' pointer, extruded into voxels (v6)
VOX = 0.075
AMB = C["amber"]
LIGHT = S3.norm([-0.35, -0.55, 0.78])


def her_xy(t):
    return HER0 + (HER1 - HER0) * ease_out(seg(t, T_PUSH, T_PUSHED), 2.2)


def scan_x(t):
    return lerp(-2.6, 2.6, seg(t, T_SCAN0, T_SCAN1))


# ------------------------------------------------------------------ surface samplers -> (points, normals)
def _basis(a):
    a = S3.norm(a)
    t = S3.norm(np.cross(a, [0, 0, 1] if abs(a[2]) < 0.9 else [1, 0, 0]))
    return a, t, np.cross(a, t)


def s_frustum(r, p0, p1, r0, r1, n):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a, t, b = _basis(p1 - p0)
    u = r.random(n)
    th = r.random(n) * 2 * math.pi
    rad = r0 + (r1 - r0) * u
    d = np.cos(th)[:, None] * t + np.sin(th)[:, None] * b
    P = p0 + (p1 - p0) * u[:, None] + d * rad[:, None]
    slope = (r0 - r1) / (np.linalg.norm(p1 - p0) + 1e-9)
    N = d + a * slope
    return P, N / np.linalg.norm(N, axis=1, keepdims=True)


def s_disk(r, c, axis, rad, n):
    a, t, b = _basis(axis)
    rr = np.sqrt(r.random(n)) * rad
    th = r.random(n) * 2 * math.pi
    P = np.asarray(c, float) + (np.cos(th) * rr)[:, None] * t + (np.sin(th) * rr)[:, None] * b
    return P, np.tile(a, (n, 1))


def s_box(r, c, s, n):
    c, s = np.asarray(c, float), np.asarray(s, float) / 2
    areas = np.array([s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2], s[0] * s[1], s[0] * s[1]])
    face = r.choice(6, n, p=areas / areas.sum())
    uv = r.uniform(-1, 1, (n, 2))
    P = np.zeros((n, 3))
    N = np.zeros((n, 3))
    for f in range(6):
        sel = face == f
        ax = f // 2
        sg = 1 if f % 2 == 0 else -1
        others = [i for i in range(3) if i != ax]
        P[sel, ax] = sg * s[ax]
        P[sel, others[0]] = uv[sel, 0] * s[others[0]]
        P[sel, others[1]] = uv[sel, 1] * s[others[1]]
        N[sel, ax] = sg
    return P + c, N


def s_ellipsoid(r, c, rad, n, yaw=0.0):
    v = r.normal(0, 1, (n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    P = v * np.asarray(rad, float)
    N = v / np.asarray(rad, float) ** 2
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return P @ R.T + np.asarray(c, float), N @ R.T


def s_tri(r, a, b, c, n):
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    u, v = r.random(n), r.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    P = a + (b - a) * u[:, None] + (c - a) * v[:, None]
    nn = S3.norm(np.cross(b - a, c - a))
    return P, np.tile(nn, (n, 1))


def toys(seed=3):
    """All the catalogued objects as one point cloud. Returns P, N, toy_id (0..3)."""
    r = np.random.default_rng(seed)
    parts = []

    def add(pn, tid):
        parts.append((pn[0], pn[1], np.full(len(pn[0]), tid)))
    # toy_01 bucket
    add(s_frustum(r, (-1.25, 0.62, 0.0), (-1.25, 0.62, 0.34), 0.17, 0.23, 2600), 0)
    add(s_disk(r, (-1.25, 0.62, 0.0), (0, 0, -1), 0.17, 300), 0)
    th = np.linspace(0, math.pi, 260)
    hp = np.stack([-1.25 + 0.23 * np.cos(th), np.full_like(th, 0.62), 0.34 + 0.16 * np.sin(th)], 1)
    add((hp, np.tile([0, -1.0, 0], (len(th), 1))), 0)
    # toy_02 castle: walls with crenellations, four towers with conical roofs, a flag
    add(s_box(r, (1.05, 0.55, 0.17), (0.72, 0.5, 0.34), 3200), 1)
    for k in range(6):
        add(s_box(r, (0.76 + k * 0.115, 0.30, 0.37), (0.06, 0.05, 0.06), 60), 1)
        add(s_box(r, (0.76 + k * 0.115, 0.80, 0.37), (0.06, 0.05, 0.06), 60), 1)
    for (x, y) in [(0.73, 0.33), (1.37, 0.33), (0.73, 0.77), (1.37, 0.77)]:
        add(s_frustum(r, (x, y, 0.0), (x, y, 0.5), 0.11, 0.11, 900), 1)
        add(s_frustum(r, (x, y, 0.5), (x, y, 0.76), 0.15, 0.0, 700), 1)
    add(s_tri(r, (1.38, 0.33, 0.95), (1.62, 0.33, 0.89), (1.38, 0.33, 0.82), 200), 1)
    add(s_frustum(r, (1.37, 0.33, 0.76), (1.37, 0.33, 0.96), 0.006, 0.006, 60), 1)
    # toy_03 spade
    add(s_box(r, (-1.05, -0.95, 0.02), (0.22, 0.28, 0.025), 900), 2)
    add(s_frustum(r, (-1.08, -0.78, 0.04), (-1.25, -0.30, 0.06), 0.025, 0.025, 500), 2)
    # toy_04 whale (the humans' plastic copy of her)
    add(s_ellipsoid(r, (0.25, -0.75, 0.16), (0.37, 0.19, 0.16), 3800, yaw=18), 3)
    cy, sy = math.cos(math.radians(18)), math.sin(math.radians(18))

    def wpt(lx, ly, lz):
        return (0.25 + lx * cy - ly * sy, -0.75 + lx * sy + ly * cy, lz)
    add(s_tri(r, wpt(-0.34, 0, 0.18), wpt(-0.62, -0.20, 0.40), wpt(-0.50, 0, 0.24), 280), 3)
    add(s_tri(r, wpt(-0.34, 0, 0.18), wpt(-0.62, 0.20, 0.40), wpt(-0.50, 0, 0.24), 280), 3)
    add(s_tri(r, wpt(0.05, -0.14, 0.10), wpt(-0.10, -0.32, 0.02), wpt(-0.08, -0.15, 0.12), 160), 3)
    add(s_tri(r, wpt(0.05, 0.14, 0.10), wpt(-0.10, 0.32, 0.02), wpt(-0.08, 0.15, 0.12), 160), 3)
    P = np.concatenate([p[0] for p in parts])
    N = np.concatenate([p[1] for p in parts])
    T = np.concatenate([p[2] for p in parts])
    return P, N, T


def hand_local(seed=5):
    """A child's hand with the index finger out. Local frame: tip at origin, finger along +a (toward the wrist)."""
    r = np.random.default_rng(seed)
    parts = []
    parts.append(s_frustum(r, (0, 0, 0.09), (0, 0, 0.95), 0.092, 0.115, 2600))
    parts.append(s_ellipsoid(r, (0, 0, 0.09), (0.092, 0.092, 0.092), 500))
    parts.append(s_ellipsoid(r, (0.03, 0, 1.25), (0.33, 0.20, 0.34), 3200))
    for k in (-1, 0, 1):
        parts.append(s_ellipsoid(r, (0.13 + 0.12 * k, 0.12, 0.95), (0.075, 0.08, 0.13), 420))
    parts.append(s_ellipsoid(r, (-0.30, 0.04, 1.05), (0.07, 0.07, 0.22), 500))
    # the nail
    nail = s_ellipsoid(r, (0, -0.075, 0.13), (0.055, 0.02, 0.07), 260)
    P = np.concatenate([p[0] for p in parts] + [nail[0]])
    N = np.concatenate([p[1] for p in parts] + [nail[1]])
    is_nail = np.zeros(len(P), bool)
    is_nail[-len(nail[0]):] = True
    return P, N, is_nail


def hand_cursor_local(v=VOX, depth=3, seed=7):
    """The pixel hand cursor (render/cursors.py) extruded into voxels; points on the exposed faces of each voxel,
    on a regular 4x4 grid plus its edges, so it reads as blocks. Local frame: x = bitmap columns, y = bitmap rows
    (row 0 = fingertip), z = depth. Returns P, N, is_outline, voxel_id."""
    import cursors
    o, f = cursors.bitmap("hand")
    filled = o | f
    h, w = filled.shape
    hc, hr = cursors.HOT["hand"]
    occ = np.zeros((h + 2, w + 2, depth + 2), bool)
    occ[1:-1, 1:-1, 1:-1] = filled[:, :, None]
    g = (np.arange(4) + 0.5) / 4 - 0.5
    gu, gv = np.meshgrid(g, g)
    grid = np.stack([gu.ravel(), gv.ravel()], 1)
    e = np.linspace(-0.5, 0.5, 6)
    edges = np.concatenate([np.stack([e, np.full(6, -0.5)], 1), np.stack([e, np.full(6, 0.5)], 1),
                            np.stack([np.full(6, -0.5), e], 1), np.stack([np.full(6, 0.5), e], 1)])
    face_uv = np.concatenate([grid, edges])
    P, N, OUT, VID = [], [], [], []
    vid = 0
    dirs = [((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1))]
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for k in range(depth):
                ctr = np.array([(c + 0.5 - hc) * v, (r + 0.5 - hr) * v, (k - (depth - 1) / 2) * v])
                for (n, a, b) in dirs:
                    # neighbour in (col, row, depth) order: n is (x, y, z) = (col, row, depth)
                    if occ[r + 1 + n[1], c + 1 + n[0], k + 1 + n[2]]:
                        continue
                    n_, a_, b_ = np.array(n, float), np.array(a, float), np.array(b, float)
                    pts = ctr + n_ * v / 2 + face_uv[:, :1] * a_ * v + face_uv[:, 1:] * b_ * v
                    P.append(pts)
                    N.append(np.tile(n_, (len(pts), 1)))
                    OUT.append(np.full(len(pts), bool(o[r, c])))
                    VID.append(np.full(len(pts), vid))
                vid += 1
    return np.concatenate(P), np.concatenate(N), np.concatenate(OUT), np.concatenate(VID), vid


def finger_tip(t):
    xy = her_xy(t)
    contact = np.array([xy[0] - 0.13, xy[1] - 0.02, 0.42 if HAND == "finger" else 0.52])
    if t < T_PUSH:
        k = smooth(seg(t, 22.3, T_PUSH))
        return contact + np.array([-0.5 * (1 - k), -0.3 * (1 - k), 3.2 * (1 - k)])
    if t < T_PUSHED + 0.1:
        return contact
    k = ease_in(seg(t, T_PUSHED + 0.1, 25.9), 2)
    return contact + np.array([-0.3 * k, -0.2 * k, 3.4 * k])


def hand_world(t, local):
    P, N, nail = local
    tip = finger_tip(t)
    a = S3.norm(np.array([-0.5, -0.9, 3.0]))           # from the tip toward the wrist
    side = S3.norm(np.cross(a, [0, 0, 1]))
    up = np.cross(side, a)
    R = np.stack([side, up, a], 1)                       # local (x, y, z) -> world
    return P @ R.T + tip, N @ R.T, nail


def cursor_basis():
    if "cbasis" not in _G:
        cam = camera(T_PUSH, (W / 2, H / 2))
        tip = finger_tip(T_PUSH)
        up0 = S3.norm(np.array([-0.18, -0.30, 1.0]))
        n = S3.norm(cam.pos - (tip + up0 * 0.7))
        up = S3.norm(up0 - n * np.dot(up0, n))
        side = S3.norm(np.cross(n, up))
        _G["cbasis"] = (side, up, n)
    return _G["cbasis"]


def cursor_world(t, local):
    P, N, out, vid, nv = local
    side, up, n = cursor_basis()
    R = np.stack([side, up, n], 1)            # local x -> side, local y (rows, away from the tip) -> up, z -> n
    return P @ R.T + finger_tip(t), N @ R.T, out, vid, nv


def sand_height(t):
    def h(X, Y):
        z = 0.025 * np.sin(X * 7.0 + 2.2 * np.sin(Y * 2.1)) + 0.012 * np.sin(Y * 11 + X * 3)
        if t > T_PUSH:
            p0, p1 = HER0, her_xy(t)
            d = p1 - p0
            L = np.linalg.norm(d) + 1e-6
            u = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (L * L), 0, 1)
            dist = np.sqrt((X - (p0[0] + u * d[0])) ** 2 + (Y - (p0[1] + u * d[1])) ** 2)
            z = z - 0.06 * np.exp(-(dist / 0.07) ** 2)
        tip = finger_tip(t)
        if tip[2] < 0.7:
            dist = np.sqrt((X - tip[0]) ** 2 + (Y - tip[1]) ** 2)
            z = z - 0.05 * np.exp(-(dist / 0.12) ** 2) * (1 - tip[2] / 0.7)
        return z
    return h


_G = {}


def _prep():
    if not _G:
        _G["toys"] = toys()
        _G["hand"] = hand_local()
        _G["hand_cur"] = hand_cursor_local()
        nv = _G["hand_cur"][4]
        _G["hand_cur_t"] = np.random.default_rng(11).uniform(22.70, 23.10, nv).astype(np.float32)
        gx, gy = np.meshgrid(np.arange(-12, 12.01, 0.45), np.arange(-6, 18.01, 0.45))
        _G["seabed"] = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.03)], 1)


def splat(buf, cam, P, col, fog=9.0, near=0.15):
    """Additive point splat. col: (n,3) or (3,) colour * brightness already applied."""
    xy, z = cam.project(P)
    ok = (z > near) & (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
    xi = xy[ok, 0].astype(np.int64)
    yi = xy[ok, 1].astype(np.int64)
    f = np.exp(-z[ok] / fog)
    col = np.asarray(col, np.float32)
    idx = yi * W + xi
    for c in range(3):
        w = (col[ok, c] if col.ndim == 2 else col[c]) * f
        buf[..., c] += np.bincount(idx, weights=w, minlength=H * W).reshape(H, W).astype(np.float32)
    return xy, z, ok


def shade(cam, P, N, base, rim=0.7):
    v = S3.norm(cam.pos - P.mean(0))
    vv = cam.pos - P
    vv /= np.linalg.norm(vv, axis=1, keepdims=True)
    lam = np.clip(N @ LIGHT, 0, 1)
    fres = (1 - np.abs(np.sum(N * vv, axis=1))) ** 2
    b = 0.22 + 0.55 * lam + rim * fres
    return b[:, None] * np.asarray(base, np.float32)[None, :]


def camera(t, center):
    k_out = ease_out(seg(t, 20.75, 22.6), 3)
    k_in = smooth(seg(t, 22.8, 23.6))
    her = np.array([*her_xy(t), 0.3])
    near_pos = np.array([-0.15, -1.55, 0.55])
    wide_pos = np.array([0.9, -5.4, 3.9])
    yaw = 0.2 * smooth(seg(t, 21.0, 26.1))
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    pos = near_pos + (R @ wide_pos - near_pos) * k_out
    tgt = np.array([HER0[0], HER0[1], 0.3]) + (np.array([0.0, 0.15, 0.0]) - np.array([HER0[0], HER0[1], 0.3])) * k_out
    pos = pos + ((pos + (her - pos) * 0.36) - pos) * k_in
    tgt = tgt + (her - tgt) * 0.6 * k_in
    return S3.Cam(pos, tgt, fov=40, center=center)


def render(fr, t, inks_body, center=(1250, 560)):
    _prep()
    cam = camera(t, center)
    xs_ = scan_x(t)
    buf = np.zeros((H, W, 3), np.float32)
    # the seabed beyond: a sparse dot field fading into the dark
    splat(buf, cam, _G["seabed"], C["grid2"] * 3.0, fog=7.0)
    # the sand: scan-lines (rows) that exist only where the sweep has passed
    rows = np.linspace(-1.44, 1.44, 36)
    xsamp = np.linspace(-1.94, 1.94, 150)
    X, Y = np.meshgrid(xsamp, rows)
    Z = sand_height(t)(X, Y)
    canvas = np.zeros((H, W), np.float32)
    head = np.zeros((H, W), np.float32)
    for j in range(len(rows)):
        sel = X[j] <= xs_
        if sel.sum() < 2:
            continue
        P = np.stack([X[j][sel], Y[j][sel], Z[j][sel]], 1)
        xy, z = cam.project(P)
        if (z < 0.2).any():
            continue
        f = float(np.exp(-z.mean() / 9.0))
        cv2.polylines(canvas, [np.round(xy * 16).astype(np.int32)], False, f, 1, cv2.LINE_AA, shift=4)
        if xs_ < 1.94:
            cv2.circle(head, tuple(np.round(xy[-1]).astype(int)), 2, 1.0, -1, cv2.LINE_AA)
    fr.fill(canvas, C["cyan"], 0.55, glow=0.25)
    # the frame: edges of the four walls (the child's; amber), drawn as the sweep passes
    segs = []
    for (x, y, sx, sy) in [(0, 1.51, 4.14, 0.12), (0, -1.51, 4.14, 0.12), (2.01, 0, 0.12, 3.14), (-2.01, 0, 0.12, 3.14)]:
        for zz in (0.0, 0.3):
            segs += [[(x - sx / 2, y - sy / 2, zz), (x + sx / 2, y - sy / 2, zz)], [(x - sx / 2, y + sy / 2, zz), (x + sx / 2, y + sy / 2, zz)],
                     [(x - sx / 2, y - sy / 2, zz), (x - sx / 2, y + sy / 2, zz)], [(x + sx / 2, y - sy / 2, zz), (x + sx / 2, y + sy / 2, zz)]]
        for (dx, dy) in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            segs.append([(x + dx * sx / 2, y + dy * sy / 2, 0.0), (x + dx * sx / 2, y + dy * sy / 2, 0.3)])
    fm = np.zeros((H, W), np.float32)
    for a, b in segs:
        a, b = np.array(a, float), np.array(b, float)
        if a[0] > xs_ and b[0] > xs_:
            continue
        if a[0] > xs_:
            a, b = b, a
        if b[0] > xs_:
            b = a + (b - a) * ((xs_ - a[0]) / (b[0] - a[0]))
        xy, z = cam.project(np.array([a, b]))
        if (z < 0.2).any():
            continue
        cv2.line(fm, tuple(np.round(xy[0] * 16).astype(int)), tuple(np.round(xy[1] * 16).astype(int)), 1.0, 1, cv2.LINE_AA, shift=4)
    fr.fill(fm, AMB, 0.85, glow=0.7)
    # toys + hand as point clouds, split around her depth for occlusion
    P, N, T = _G["toys"]
    vis = P[:, 0] <= xs_
    flash = np.exp(-((P[:, 0] - xs_) / 0.06) ** 2)
    colT = shade(cam, P, N, AMB * 0.75) * vis[:, None] + flash[:, None] * vis[:, None] * C["ice"] * 1.5
    if HAND == "finger":
        hp, hn, nail = hand_world(t, _G["hand"])
        show_hand = finger_tip(t)[2] < 3.6
        colH = shade(cam, hp, hn, hexc("#ff9a46") * 0.8, rim=0.9)
        colH[nail] = C["ice"] * 0.9
    else:
        hp, hn, outl, vid, _nv = cursor_world(t, _G["hand_cur"])
        appear = _G["hand_cur_t"][vid]
        on = t >= appear
        age = np.maximum(t - appear, 0)
        colH = shade(cam, hp, hn, AMB * 0.55, rim=0.5)
        colH[outl] = shade(cam, hp[outl], hn[outl], AMB * 1.25, rim=0.9)
        colH = colH * on[:, None] + (np.exp(-age * 20) * on)[:, None] * C["ice"] * 2.0
        hp, colH = hp[on], colH[on]
        show_hand = finger_tip(t)[2] < 3.6 and len(hp) > 0
    her3 = np.array([*her_xy(t), 0.3])
    zher = cam.cam(her3[None])[0, 2]
    zT = cam.cam(P)[:, 2]
    zH = cam.cam(hp)[:, 2]
    back = np.zeros((H, W, 3), np.float32)
    front = np.zeros((H, W, 3), np.float32)
    splat(back, cam, P[zT > zher], colT[zT > zher] * 1.7)
    splat(front, cam, P[zT <= zher], colT[zT <= zher] * 1.7)
    if show_hand:
        splat(back, cam, hp[zH > zher], colH[zH > zher] * 1.1)
        splat(front, cam, hp[zH <= zher], colH[zH <= zher] * 1.1)
    for b in (buf, back):
        g = cv2.GaussianBlur(b, (0, 0), 0.75) * 1.6
        fr.img += np.clip(g, 0, 1.0)
        fr.glow += np.clip(g, 0, 2.0) * 0.45
    # her (the only solid thing here): contact shadow + the printed figure; 'regenerate' swaps her
    feet_xy, zf = cam.project(np.array([[*her_xy(t), 0.0]]))
    feet = feet_xy[0]
    hpx = cam.f * HER_H / zf[0]
    rot = -24 * ease_out(seg(t, T_PUSH, T_PUSHED + 0.3), 2)
    gone = smooth(seg(t, T_REGEN, T_REGEN + 0.35))
    her_box = None
    if gone < 1:
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(feet[0]), int(feet[1])), (int(hpx * 0.26), int(hpx * 0.07)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        fr.fill(cv2.GaussianBlur(sh, (0, 0), 3), C["ink"], 0.7 * (1 - gone))
        Pf = inks_body.warp(CHAR["feet"], (feet[0], feet[1]), hpx / (CHAR["feet"][1] - CHAR["top"][1]), rot)
        if gone > 0:
            Pf = Pf * dots(H, W, np.full((H, W), 1 - gone, np.float32), 6, 45)[..., None]
        print_figure(fr, Pf, shadow_off=(6, 4))
        her_box = (feet, hpx, "toy_05", gone)
    if t >= T_REGEN + 0.12:
        k = ease_back(seg(t, T_REGEN + 0.12, T_REGEN + 0.34), 2.0)
        p2n, zn = cam.project(np.array([[HER0[0], HER0[1], 0.0]]))
        feet2 = p2n[0]
        hpx2 = cam.f * HER_H / zn[0]
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(feet2[0]), int(feet2[1])), (int(hpx2 * 0.26), int(hpx2 * 0.07)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        fr.fill(cv2.GaussianBlur(sh, (0, 0), 3), C["ink"], 0.7 * min(1, k))
        Pn = inks_body.warp(CHAR["feet"], (feet2[0], feet2[1]), max(0.02, k) * hpx2 / (CHAR["feet"][1] - CHAR["top"][1]), 0)
        print_figure(fr, Pn, shadow_off=(6, 4))
        her_box = (feet2, hpx2, "toy_06", 0.0)
    g = cv2.GaussianBlur(front, (0, 0), 0.75) * 1.6
    fr.img += np.clip(g, 0, 1.0)
    fr.glow += np.clip(g, 0, 2.0) * 0.45
    # the sweep itself: a translucent vertical plane
    if T_SCAN0 <= t <= T_SCAN1 + 0.05:
        quad = np.array([[xs_, -1.7, 0.0], [xs_, 1.7, 0.0], [xs_, 1.7, 1.1], [xs_, -1.7, 1.1]])
        xy, z = cam.project(quad)
        if (z > 0.2).all():
            pm = np.zeros((H, W), np.float32)
            cv2.fillConvexPoly(pm, np.round(xy * 16).astype(np.int32), 1.0, cv2.LINE_AA, shift=4)
            fr.fill(pm, C["cyan"], 0.08, glow=0.05)
            pl = np.zeros((H, W), np.float32)
            cv2.polylines(pl, [np.round(xy * 16).astype(np.int32)], True, 1.0, 1, cv2.LINE_AA, shift=4)
            fr.fill(pl, C["cyan"], 0.9, glow=0.9)
        fr.fill(cv2.GaussianBlur(head, (0, 0), 1.0), C["ice"], 1.0, glow=1.0)
    return cam, her_box


DETECT = [(21.30, 0, "toy_01  bucket  0.98"), (21.66, 1, "toy_02  castle  0.97"),
          (22.01, 2, "toy_03  spade  0.95"), (22.37, 3, "toy_04  whale  0.91")]


def _det_box(fr, t, t0, x0, y0, x1, y1, txt, col, amt=1.0):
    p = seg(t, t0, t0 + 0.25)
    if p <= 0 or amt <= 0:
        return
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = ease_back(p, 1.6)
    x0, x1 = cx + (x0 - cx) * (1.25 - 0.25 * k), cx + (x1 - cx) * (1.25 - 0.25 * k)
    y0, y1 = cy + (y0 - cy) * (1.25 - 0.25 * k), cy + (y1 - cy) * (1.25 - 0.25 * k)
    m = np.zeros((H, W), np.float32)
    L = 14
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(m, (int(x), int(y)), (int(x + sx * L), int(y)), 1.0, 2, cv2.LINE_AA)
        cv2.line(m, (int(x), int(y)), (int(x), int(y + sy * L)), 1.0, 2, cv2.LINE_AA)
    cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 0.35, 1, cv2.LINE_AA)
    fr.fill(m, col, amt * min(1, p * 3), glow=0.5)
    s = decode(txt, t, t0 + 0.05, 0.35)
    if s:
        tm_w = 11 * len(s) + 16
        tag = np.zeros((H, W), np.float32)
        cv2.rectangle(tag, (int(x0), int(y0 - 26)), (int(x0 + tm_w), int(y0 - 2)), 1.0, -1)
        fr.fill(tag, col, 0.85 * amt)
        label(fr, s, x0 + 6, y0 - 14, C["bg0"], 16, amt, "lc")


def detections(fr, t, cam, her_box):
    P, _, T = _G["toys"]
    xy, z = cam.project(P)
    for (t0, tid, txt) in DETECT:
        if t < t0:
            continue
        sel = (T == tid) & (z > 0.2)
        if sel.sum() < 10:
            continue
        x0, y0 = np.percentile(xy[sel], 1, axis=0) - 8
        x1, y1 = np.percentile(xy[sel], 99, axis=0) + 8
        _det_box(fr, t, t0, x0, y0, x1, y1, txt, C["cyan"])
    if her_box is not None and t >= 22.72:
        feet, hpx, name, gone = her_box
        x0, x1 = feet[0] - hpx * 0.32, feet[0] + hpx * 0.32
        y0, y1 = feet[1] - hpx * 1.02, feet[1] + 6
        blink = 1.0 if (t % 0.4) < 0.28 else 0.55
        t0 = 22.72 if name == "toy_05" else T_REGEN + 0.3
        _det_box(fr, t, t0, x0, y0, x1, y1, f"{name}  ?  0.51", C["amber"], blink * (1 - gone))
