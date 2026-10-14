"""The chat client, drawn in code after the user's own DeepSeek desktop client (assets/from_user/ui_ref/client_page_*.png),
customised for the film: orange = the user's messages, her replies in serif, her emblem as the avatar,
the thinking line '深度求索中，用时 N 秒', the status bar's cache-hit rate as her memory, the desk pet in the corner."""
import math
from functools import lru_cache

import cv2
import numpy as np

from lib import hexc
from plates import INK, Emblem, Inks, fill, put, text
from build import decode

PLAIN = "SAME"          # chat text decodes through look-alike hanzi / same-class letters (build.wrong_char)

UI_PAPER = dict(bg=hexc("#efeae0"), dot=hexc("#d8d1c3"), ink=hexc("#171b26"), sub=hexc("#8b909c"), line=hexc("#ddd6c8"),
                box=hexc("#fbfaf6"), box_line=hexc("#d6cfbf"), chip=hexc("#e4dfd4"), you=INK["orange"], you_ink=hexc("#1a0d06"),
                you_fill=INK["orange"], you_line=INK["orange"], her=INK["blue"], think=hexc("#3156e0"), typed=hexc("#171b26"),
                pet=(hexc("#171b26"), hexc("#8c93a8"), hexc("#fbf8f2")))
UI_CYBER = dict(bg=hexc("#0a1124"), dot=hexc("#18284d"), ink=hexc("#d7e3f7"), sub=hexc("#7884a6"), line=hexc("#1d2e55"),
                box=hexc("#0e1834"), box_line=hexc("#2a4c86"), chip=hexc("#16264d"), you=hexc("#ffb23e"), you_ink=hexc("#ffcf86"),
                you_fill=hexc("#1c1408"), you_line=hexc("#ffb23e"), her=hexc("#3ee6ff"), think=hexc("#3ee6ff"), typed=hexc("#ffcf86"),
                pet=(hexc("#060a18"), hexc("#3a5694"), hexc("#dff4ff")))
UI = dict(UI_PAPER)


def set_theme(name):
    UI.clear()
    UI.update(UI_CYBER if name == "cyber" else UI_PAPER)
_C = {}


@lru_cache(maxsize=1024)
def T(s, size, role="sans"):
    return text(s, size, role)


def emblem():
    if "emb" not in _C:
        _C["emb"] = Emblem()
    return _C["emb"]


def pet():
    if "pet" not in _C:
        from lib import ROOT
        _C["pet"] = Inks(scale=1.0, faceless=False, path=ROOT / "assets" / "from_user" / "ui_ref" / "desk_pet_cutout.png",
                         min_k=0.25, t1=0.33, t2=0.74).plates
    return _C["pet"]


def rrect(m, x0, y0, x1, y1, r, val=1.0):
    x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), val, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), val, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, val, -1, cv2.LINE_AA)
    return m


def rrect_line(shape, x0, y0, x1, y1, r, w=1.5):
    a = rrect(np.zeros(shape, np.float32), x0, y0, x1, y1, r)
    b = rrect(np.zeros(shape, np.float32), x0 + w, y0 + w, x1 - w, y1 - w, max(1, r - w))
    return np.clip(a - b, 0, 1)


def dot_grid(h, w, step=22):
    key = ("grid", h, w, step)
    if key not in _C:
        m = np.zeros((h, w), np.float32)
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                cv2.circle(m, (x * 4, y * 4), 5, 1.0, -1, cv2.LINE_AA, shift=2)
        _C[key] = m
    return _C[key]


def spinner(shape, cx, cy, r, t):
    m = np.zeros(shape, np.float32)
    for k in range(6):
        a = t * 3.0 + k * math.pi / 3
        cv2.line(m, (int(cx + math.cos(a) * r * 0.35), int(cy + math.sin(a) * r * 0.35)),
                 (int(cx + math.cos(a) * r), int(cy + math.sin(a) * r)), 1.0, 2, cv2.LINE_AA)
    return m


def composer(c, t, x0, y0, x1, y1, typed="", caret=True, placeholder="发消息或创建任务，/ 调用指令，@ 文件或对话"):
    sh = c.shape[:2]
    c = fill(c, rrect(np.zeros(sh, np.float32), x0, y0 + 4, x1, y1 + 4, 26), UI["line"])
    c = fill(c, rrect(np.zeros(sh, np.float32), x0, y0, x1, y1, 26), UI["box"])
    c = fill(c, rrect_line(sh, x0, y0, x1, y1, 26, 1.5), UI["box_line"])
    tx, ty = x0 + 28, y0 + 34
    if typed:
        tm = T(typed, 26, "sans")
        c = put(c, tm, tx, ty, UI["typed"], 1.0, "lc")
        cx_ = tx + tm.shape[1] + 4
    else:
        c = put(c, T(placeholder, 24, "sans"), tx, ty, UI["sub"], 1.0, "lc")
        cx_ = tx - 2
    if caret and (caret == "on" or (caret != "off" and (t % 0.6) < 0.36)):
        m = np.zeros(sh, np.float32)
        cv2.rectangle(m, (int(cx_), ty - 16), (int(cx_) + 3, ty + 16), 1.0, -1)
        c = fill(c, m, UI["you"])
    by = y1 - 30
    m = np.zeros(sh, np.float32)
    cv2.circle(m, (x0 + 34, by), 19, 1.0, -1, cv2.LINE_AA)
    c = fill(c, m, UI["chip"])
    m = np.zeros(sh, np.float32)
    cv2.line(m, (x0 + 26, by), (x0 + 42, by), 1.0, 2, cv2.LINE_AA)
    cv2.line(m, (x0 + 34, by - 8), (x0 + 34, by + 8), 1.0, 2, cv2.LINE_AA)
    c = fill(c, m, UI["ink"])
    m = np.zeros(sh, np.float32)
    sx = x0 + 78
    cv2.polylines(m, [np.array([[sx, by - 9], [sx + 7, by - 6], [sx + 14, by - 9], [sx + 14, by + 1], [sx + 7, by + 9], [sx, by + 1]], np.int32)], True, 1.0, 1, cv2.LINE_AA)
    c = fill(c, m, UI["sub"])
    c = put(c, T("完全权限", 20, "sans"), sx + 22, by, UI["sub"], 1.0, "lc")
    c = put(c, T("DeepSeek-V41-Flash  Max", 20, "sans"), x1 - 30, by, UI["sub"], 1.0, "rc")
    return c


def client_canvas(w, h, t, *, title="新对话", msgs=(), typed="", caret=True, status="", scroll=0.0, pet_at=None,
                  pet_k=1.0, composer_y=None, thread_r=980, pet_t0=None):
    """Draw the client into a (h, w, 3) canvas. msgs: dicts with who ('you'/'her'), at, text, and for her: think (start)."""
    c = np.broadcast_to(UI["bg"], (h, w, 3)).astype(np.float32).copy()
    sh = (h, w)
    c = fill(c, dot_grid(h, w), UI["dot"])
    # window chrome
    m = np.zeros(sh, np.float32)
    rrect(m, 40, 18, 62, 40, 4)
    m = np.clip(m - rrect(np.zeros(sh, np.float32), 42, 20, 60, 38, 3), 0, 1)
    cv2.line(m, (48, 20), (48, 38), 1.0, 2)
    cv2.circle(m, (95, 29), 11, 1.0, 2, cv2.LINE_AA)
    cv2.line(m, (90, 29), (100, 29), 1.0, 2)
    cv2.line(m, (95, 24), (95, 34), 1.0, 2)
    for k, x in enumerate((w - 170, w - 115, w - 60)):
        if k == 0:
            cv2.line(m, (x - 8, 29), (x + 8, 29), 1.0, 2)
        elif k == 1:
            cv2.rectangle(m, (x - 8, 21), (x + 8, 37), 1.0, 2)
        else:
            cv2.line(m, (x - 8, 21), (x + 8, 37), 1.0, 2, cv2.LINE_AA)
            cv2.line(m, (x - 8, 37), (x + 8, 21), 1.0, 2, cv2.LINE_AA)
    c = fill(c, m, UI["ink"] * 0.8)
    c = put(c, T("应用", 22), 140, 29, UI["ink"], 0.85, "lc")
    c = put(c, T("编辑", 22), 210, 29, UI["ink"], 0.85, "lc")
    # header: title (auto-named after the first message) + mode chip
    tm = T(title, 30, "sans")
    c = put(c, tm, 40, 86, UI["ink"], 1.0, "lc")
    cx = 40 + tm.shape[1] + 24
    m = np.zeros(sh, np.float32)
    cv2.circle(m, (cx + 8, 86), 8, 1.0, 2, cv2.LINE_AA)
    c = fill(c, m, UI["sub"])
    c = put(c, T("标准模式", 22), cx + 24, 86, UI["sub"], 1.0, "lc")
    # tabs
    c = put(c, T("对话", 24), 40, 132, UI["ink"], 1.0, "lc")
    c = put(c, T("轨迹", 24), 130, 132, UI["sub"], 1.0, "lc")
    m = np.zeros(sh, np.float32)
    cv2.rectangle(m, (40, 152), (88, 155), 1.0, -1)
    c = fill(c, m, UI["her"])
    m = np.zeros(sh, np.float32)
    cv2.line(m, (0, 160), (w, 160), 1.0, 1)
    c = fill(c, m, UI["line"])
    # thread
    th = np.broadcast_to(UI["bg"], (h, w, 3)).astype(np.float32).copy()
    th = fill(th, dot_grid(h, w), UI["dot"])
    y = int(round(200 - scroll))
    for msg in msgs:
        if msg["who"] == "you":
            if t < msg["at"]:
                continue
            tm_full = T(msg["text"], 26, "sans")
            bx1 = msg.get("right", thread_r)
            bx0 = bx1 - tm_full.shape[1] - 44
            age = t - msg["at"]
            # construction: the outline sweeps in from the tail side, the fill follows behind a bright edge,
            # then the words decode
            pe = min(1.0, age / 0.16)
            pf = min(1.0, max(0.0, (age - 0.06) / 0.18))
            xs = np.arange(sh[1], dtype=np.float32)[None, :]
            sweep_e = (xs >= bx1 - (bx1 - bx0 + 24) * pe).astype(np.float32)
            sweep_f = (xs >= bx1 - (bx1 - bx0 + 24) * pf).astype(np.float32)
            mm = rrect(np.zeros(sh, np.float32), bx0, y, bx1, y + 56, 20)
            th = fill(th, mm * sweep_f, UI["you_fill"])
            edge = np.clip(mm - rrect(np.zeros(sh, np.float32), bx0 + 1.5, y + 1.5, bx1 - 1.5, y + 54.5, 18.5), 0, 1)
            th = fill(th, edge * sweep_e, UI["you_line"])
            if pf < 1:
                xf = int(bx1 - (bx1 - bx0 + 24) * pf)
                sl = np.zeros(sh, np.float32)
                cv2.line(sl, (xf, y - 3), (xf, y + 59), 1.0, 2)
                th = fill(th, sl * mm.max(), UI["you_line"])
            txt = decode(msg["text"], t, msg["at"] + 0.14, 0.22, ascii_pool=PLAIN)
            if txt:
                th = put(th, T(txt, 26, "sans"), bx0 + 22, y + 28, UI["you_ink"], 1.0, "lc")
            tl = decode(msg.get("time", "06:13"), t, msg["at"] + 0.3, 0.15, ascii_pool=PLAIN)
            if tl:
                th = put(th, T(tl, 16, "mono"), bx1 - 4, y + 76, UI["sub"], 1.0, "rc")
            y += 108
        else:
            think = msg.get("think", msg["at"] - 0.8)
            if t < think:
                continue
            # avatar: her emblem
            e = emblem()
            th = e.stamp(th, 62, y + 14, 30, UI["her"], UI["bg"])
            if t < msg["at"]:
                secs = int(t - think) + 1
                th = fill(th, spinner(sh, 100, y + 14, 9, t), UI["think"])
                th = put(th, T(f"深度求索中，用时 {secs} 秒 …", 20), 118, y + 14, UI["think"], 1.0, "lc")
                y += 60
            else:
                dur = msg.get("dur", max(1, int(round(msg["at"] - think))))
                th = put(th, T(f"已完成，用时 {dur} 秒", 20), 92, y + 14, UI["sub"], 1.0, "lc")
                m = np.zeros(sh, np.float32)
                cv2.line(m, (92, y + 36), (min(w - 40, thread_r + 20), y + 36), 1.0, 1)
                th = fill(th, m, UI["line"])
                n = len(msg["text"])
                txt = msg["text"] if msg.get("instant") else decode(msg["text"], t, msg["at"], n * 0.045 + 0.1, head=3,
                                                                       ascii_pool=PLAIN)
                shown = len(txt) if txt == msg["text"] else 0
                rows = msg["text"].count("\n")                 # her answer may run to more than one line
                if txt:
                    for k, part in enumerate(txt.split("\n")):
                        if part:
                            th = put(th, T(part, 30, "serif"), 92, y + 72 + 46 * k, UI["ink"], 1.0, "lc")
                if shown == n:
                    th = put(th, T(msg.get("usage", "用量 0.4K tok"), 16, "sans"), 92, y + 112 + 46 * rows, UI["sub"], 1.0, "lc")
                y += 140 + 46 * rows
    cy0 = composer_y if composer_y is not None else h - 150
    clip = np.zeros(sh, np.float32)
    cv2.rectangle(clip, (0, 161), (w, cy0 - 6), 1.0, -1)
    c = c * (1 - clip[..., None]) + th * clip[..., None]
    # composer + status bar at the foot
    c = composer(c, t, 40, cy0, w - 40, cy0 + 104, typed, caret)
    if status:
        c = put(c, T(status, 18, "sans"), w // 2, cy0 + 126, UI["sub"], 1.0, "cc")
    # the desk pet, peeking over the composer from the corner
    if pet_at is not None:
        P = pet()
        ph, pw = P.shape[:2]
        s = pet_k
        M = np.float32([[s, 0, pet_at[0] - pw * s], [0, s, pet_at[1] - ph * s]])
        Pw = cv2.warpAffine(P, M, (w, h), flags=cv2.INTER_LINEAR)
        layer = c.copy()
        for mm, col in ((Pw[..., 1], UI["pet"][1]), (Pw[..., 2], UI["pet"][2]), (Pw[..., 0], UI["pet"][0])):
            layer = fill(layer, mm, col)
        a = Pw[..., 3:4]
        if pet_t0 is not None and t < pet_t0 + 0.5:
            # rez in: big pixels resolving to sharp
            from build import mosaic_in
            x0p = int(max(0, pet_at[0] - pw * s))
            y0p = int(max(0, pet_at[1] - ph * s))
            patch = mosaic_in(np.concatenate([layer[y0p:, x0p:], a[y0p:, x0p:]], 2), t, pet_t0, 0.45)
            if patch is None:
                return c
            layer[y0p:, x0p:] = patch[..., :3]
            a = a.copy()
            a[y0p:, x0p:] = (patch[..., 3:4] > 0.35).astype(np.float32)
        c = c * (1 - a) + layer * a
    return c
