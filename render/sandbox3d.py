"""A tiny software 3D renderer (numpy + OpenCV, painter's algorithm) for the S06 sandbox. No Blender.

World: x right, y away from the camera, z up. One unit ~ one metre of a (giant) sandbox.
"""
import math

import cv2
import numpy as np

from lib import H, W


def norm(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


class Cam:
    def __init__(self, pos, target, fov=38.0, center=(W / 2, H / 2), up=(0, 0, 1)):
        self.pos = np.asarray(pos, np.float64)
        f = norm(np.asarray(target, np.float64) - self.pos)
        r = norm(np.cross(f, up))
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])
        self.f = (H / 2) / math.tan(math.radians(fov / 2))
        self.c = center

    def cam(self, P):
        return (np.asarray(P, np.float64) - self.pos) @ self.R.T

    def project(self, P):
        Q = self.cam(P)
        z = np.maximum(Q[..., 2], 1e-6)
        x = self.c[0] + self.f * Q[..., 0] / z
        y = self.c[1] - self.f * Q[..., 1] / z
        return np.stack([x, y], -1), Q[..., 2]


class Mesh:
    def __init__(self, V, F, color, edge=None, edge_amt=0.8, glow=0.0, name=""):
        self.V = np.asarray(V, np.float64)
        self.F = F
        self.color = np.asarray(color, np.float64)
        self.edge = None if edge is None else np.asarray(edge, np.float64)
        self.edge_amt, self.glow, self.name = edge_amt, glow, name

    def centroid(self):
        return self.V.mean(0)


# ------------------------------------------------------------------ primitives
def box(c, s, color, **kw):
    cx, cy, cz = c
    sx, sy, sz = (np.asarray(s) / 2)
    V = [(cx + dx * sx, cy + dy * sy, cz + dz * sz) for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return Mesh(V, F, color, **kw)


def frustum(p0, p1, r0, r1, color, seg=18, caps=True, **kw):
    """Truncated cone between points p0 (radius r0) and p1 (radius r1)."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    a = norm(p1 - p0)
    t = norm(np.cross(a, [0, 0, 1] if abs(a[2]) < 0.9 else [1, 0, 0]))
    b = np.cross(a, t)
    V, F = [], []
    for k in range(seg):
        th = 2 * math.pi * k / seg
        d = t * math.cos(th) + b * math.sin(th)
        V.append(p0 + d * r0)
        V.append(p1 + d * r1)
    for k in range(seg):
        i, j = 2 * k, 2 * ((k + 1) % seg)
        F.append([i, j, j + 1, i + 1])
    if caps:
        F.append([2 * k for k in range(seg)][::-1])
        if r1 > 1e-6:
            F.append([2 * k + 1 for k in range(seg)])
    return Mesh(V, F, color, **kw)


def ellipsoid(c, r, color, nu=16, nv=10, yaw=0.0, **kw):
    c = np.asarray(c, np.float64)
    V, F = [], []
    cy_, sy_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    for i in range(nv + 1):
        ph = math.pi * i / nv - math.pi / 2
        for j in range(nu):
            th = 2 * math.pi * j / nu
            x, y, z = r[0] * math.cos(ph) * math.cos(th), r[1] * math.cos(ph) * math.sin(th), r[2] * math.sin(ph)
            V.append(c + np.array([x * cy_ - y * sy_, x * sy_ + y * cy_, z]))
    for i in range(nv):
        for j in range(nu):
            a, b = i * nu + j, i * nu + (j + 1) % nu
            F.append([a, b, b + nu, a + nu])
    return Mesh(V, F, color, **kw)


def flat_poly(pts, color, **kw):
    return Mesh(pts, [list(range(len(pts)))], color, **kw)


# ------------------------------------------------------------------ renderer
LIGHT = norm([-0.35, -0.55, 0.78])


def shade(n, base, levels=(0.38, 0.62, 0.85, 1.0)):
    l = max(0.0, float(np.dot(n, LIGHT)))
    q = levels[min(len(levels) - 1, int(l * len(levels)))]
    return base * q


def draw_meshes(canvas, glow, cam, meshes, near=0.15):
    """Painter's algorithm over all faces of all meshes. canvas/glow: uint8 BGR? No: RGB uint8 arrays."""
    faces = []
    for mi, m in enumerate(meshes):
        P2, Z = cam.project(m.V)
        Qc = cam.cam(m.V)
        for f in m.F:
            if np.any(Z[f] < near):
                continue
            v = m.V[f]
            n = norm(np.cross(v[1] - v[0], v[2] - v[0])) if len(f) >= 3 else np.array([0, 0, 1.0])
            ctr = v.mean(0)
            if np.dot(n, ctr - cam.pos) > 0:          # back face
                continue
            faces.append((float(Z[f].mean()), mi, P2[f], n))
    faces.sort(key=lambda x: -x[0])
    for _, mi, p2, n in faces:
        m = meshes[mi]
        col = shade(n, m.color)
        pts = np.round(p2 * 16).astype(np.int32)
        cv2.fillConvexPoly(canvas, pts, tuple(float(c) * 255 for c in col), cv2.LINE_AA, shift=4)
        if m.edge is not None:
            ec = tuple(float(c) * 255 * m.edge_amt for c in m.edge)
            cv2.polylines(canvas, [pts], True, ec, 1, cv2.LINE_AA, shift=4)
            if m.glow:
                cv2.polylines(glow, [pts], True, tuple(float(c) * 255 * m.glow for c in m.edge), 1, cv2.LINE_AA, shift=4)


def draw_lines(canvas, cam, segs, color, amt=1.0, fade=None, near=0.2, thick=1, glow=None, glow_amt=0.0):
    """segs: (n, 2, 3) world segments. fade(z) -> brightness multiplier."""
    P, Z = cam.project(segs.reshape(-1, 3))
    P = P.reshape(-1, 2, 2)
    Z = Z.reshape(-1, 2)
    for k in range(len(segs)):
        if Z[k].min() < near:
            continue
        a = amt * (fade(Z[k].mean()) if fade else 1.0)
        if a < 0.02:
            continue
        p = np.round(P[k] * 16).astype(np.int32)
        cv2.line(canvas, tuple(p[0]), tuple(p[1]), tuple(float(c) * 255 * a for c in color), thick, cv2.LINE_AA, shift=4)
        if glow is not None and glow_amt:
            cv2.line(glow, tuple(p[0]), tuple(p[1]), tuple(float(c) * 255 * a * glow_amt for c in color), thick, cv2.LINE_AA, shift=4)


def terrain(x0, x1, y0, y1, nx, ny, hfun):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = hfun(X, Y)
    V = np.stack([X, Y, Z], -1)
    return V


def draw_terrain(canvas, glow, cam, V, base, line_col, line_amt=0.35):
    """Height-field: shaded quads far->near, then the grid lines on top."""
    ny, nx, _ = V.shape
    P, Z = cam.project(V.reshape(-1, 3))
    P = P.reshape(ny, nx, 2)
    Z = Z.reshape(ny, nx)
    quads = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            z = (Z[j, i] + Z[j, i + 1] + Z[j + 1, i] + Z[j + 1, i + 1]) / 4
            quads.append((z, j, i))
    quads.sort(key=lambda q: -q[0])
    for z, j, i in quads:
        v = V[[j, j, j + 1, j + 1], [i, i + 1, i + 1, i]]
        n = norm(np.cross(v[1] - v[0], v[3] - v[0]))
        col = shade(n, base, levels=(0.55, 0.7, 0.85, 1.0))
        pts = np.round(P[[j, j, j + 1, j + 1], [i, i + 1, i + 1, i]] * 16).astype(np.int32)
        cv2.fillConvexPoly(canvas, pts, tuple(float(c) * 255 for c in col), cv2.LINE_AA, shift=4)
    lc = tuple(float(c) * 255 * line_amt for c in line_col)
    for j in range(ny):
        cv2.polylines(canvas, [np.round(P[j] * 16).astype(np.int32)], False, lc, 1, cv2.LINE_AA, shift=4)
    for i in range(nx):
        cv2.polylines(canvas, [np.round(P[:, i] * 16).astype(np.int32)], False, lc, 1, cv2.LINE_AA, shift=4)
