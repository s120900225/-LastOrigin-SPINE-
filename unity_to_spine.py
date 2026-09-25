#!/usr/bin/env python3
"""Export a Unity 2D rigged character AssetBundle to Spine 3.8.

Usage::

    python unity_to_spine.py path/to/__data
    python unity_to_spine.py path/to/__data --output path/to/spine_editor
    python unity_to_spine.py path/to/__data --runtime
    python unity_to_spine.py path/to/__data --gif
    python unity_to_spine.py path/to/__data --gif-only --gif-width 720

Pass the bundle file or directory that UnityPy accepts (typically ``__data``).
Outputs ``skeleton.json`` + ``skeleton.atlas`` (+ texture pages) under
``<bundle-parent>/spine_editor/`` by default (Spine Editor import layout),
or ``--output`` / ``--runtime`` for the runtime layout.
With ``--gif``, each ``AnimationClip`` is also rasterised to
``<bundle-parent>/gifs/anim_<name>.gif``.

This script is self-contained: it reads the Unity bundle, decodes skinned
meshes / sprites / animation clips, and writes a Spine skeleton.  It does not
import :mod:`assemble` or :mod:`to_spine`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import zlib
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import UnityPy
from PIL import Image
from UnityPy.helpers.MeshHelper import MeshHandler

SPINE_VERSION = "3.8.99"
FPS = int(os.environ.get("SPINE_FPS", "30"))
TARGET_W = int(os.environ.get("SPINE_TARGET_W", "1600"))
PAD = 0.04
GIF_W = int(os.environ.get("GIF_W", "720"))
GIF_FPS = int(os.environ.get("GIF_FPS", "24"))
GIF_BG = os.environ.get("GIF_BG", "ffffff")
GIF_WORKERS = int(os.environ.get("GIF_WORKERS", "0"))  # 0 = auto (cpu count, cap 8)
EPS = 1e-4
ATLAS_SUFFIX_RE = re.compile(r"(parts?\d+)$", re.I)

# Unity AnimationClip generic binding IDs (Transform=4, GameObject=1, SMR=137).
GO_TYPE = 1
GO_ACTIVE_ATTR = 2086281974
SMR_TYPE = 137
SMR_COLOR_A_ATTR = 2108656497
SMR_BLEND_CROSSFADE = 2484067052
# SkinnedMeshRenderer customType 22 weight (Balgure blush fade in Lobby/BreastTouch).
SMR_BLEND_SHAPE_WEIGHT = 2274245065
MOUTH_IDLE_PART = "Part_Mouth_Idle_41"
MOUTH_A_PART = "Part_Mouth_A_41"
BALGURE_PARTS = frozenset({"Part_Balgure_L_41", "Part_Balgure_R_41"})
VIS_EPS = 0.01


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------
def mat4(m) -> np.ndarray:
    return np.array(
        [[getattr(m, f"e{r}{c}") for c in range(4)] for r in range(4)],
        dtype=np.float64,
    )


def quat_to_m3(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_world(TR, overrides=None):
    """Local->world matrices; optional per-bone overrides without copying TR."""
    overrides = overrides or {}
    wcache = {}

    def trs(pid):
        t = TR[pid]
        o = overrides.get(pid)
        if not o:
            return t["pos"], t["rot"], t["scale"]
        return (
            o.get("pos", t["pos"]),
            o.get("rot", t["rot"]),
            o.get("scale", t["scale"]),
        )

    def world(pid):
        if pid in wcache:
            return wcache[pid]
        chain = []
        q = pid
        while q and q in TR:
            chain.append(q)
            q = TR[q]["father"]
        M = np.eye(4)
        for q in reversed(chain):
            pos, rot, scale = trs(q)
            T = np.eye(4)
            T[:3, 3] = pos
            R = np.eye(4)
            R[:3, :3] = quat_to_m3(rot)
            S = np.diag([*scale, 1.0])
            M = M @ T @ R @ S
        wcache[pid] = M
        return M

    return world


def local_matrix(pos, rot, scale):
    T = np.eye(4)
    T[:3, 3] = pos
    R = np.eye(4)
    R[:3, :3] = quat_to_m3(rot)
    S = np.diag([scale[0], scale[1], scale[2], 1.0])
    return T @ R @ S


def norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def decompose2d(M):
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    tx, ty = M[0, 3], M[1, 3]
    rotation = math.degrees(math.atan2(c, a))
    scaleX = math.hypot(a, c)
    rotationY = math.degrees(math.atan2(d, b))
    scaleY = math.hypot(b, d)
    shearY = norm180(rotationY - 90.0 - rotation)
    return dict(
        x=float(tx), y=float(ty), rotation=float(rotation),
        scaleX=float(scaleX), scaleY=float(scaleY), shearY=float(shearY),
    )


def trs_to_spine(t):
    return decompose2d(local_matrix(t["pos"], t["rot"], t["scale"]))


def r2(v):
    return round(float(v), 4)


def atlas_key_from_name(name: str | None) -> str | None:
    if not name:
        return None
    m = ATLAS_SUFFIX_RE.search(name)
    return m.group(1).lower() if m else None


def skin_part(part, world):
    if part["kind"] == "sprite":
        M = world(part["tr_pid"])
        return (M @ part["sv2"].T).T[:, :2]

    bones, bind, vh = part["bones"], part["bind"], part["vh"]
    if part["single"]:
        if len(bones) == 0:
            # No bones – return vertices as-is
            return vh[:, :2]
        M = world(bones[0]) @ bind[0]
        return (M @ vh.T).T[:, :2]

    bi, bw = part["bi"], part["bw"]
    vc, k = bi.shape
    BW = [world(b) @ bind[i] for i, b in enumerate(bones)]
    pos = np.zeros((vc, 3))
    for ki in range(k):
        w = bw[:, ki]
        if not np.any(w):
            continue
        for bidx in np.unique(bi[:, ki]):
            sel = bi[:, ki] == bidx
            pts = (BW[bidx] @ vh[sel].T).T[:, :3]
            pos[sel] += pts * w[sel, None]
    return pos[:, :2]


def skin_all(parts, world):
    out = []
    for p in parts:
        try:
            out.append(skin_part(p, world))
        except (KeyError, IndexError):
            # Same malformed bone/bind data build_slots_and_skin() also
            # skips for this part - excluded from the bounds calc too.
            pass
    return out


def bounds_of(positions, pad=0.0):
    allxy = np.vstack(positions)
    minx, miny = allxy.min(0)
    maxx, maxy = allxy.max(0)
    w, h = maxx - minx, maxy - miny
    return (minx - w * pad, miny - h * pad, maxx + w * pad, maxy + h * pad)


def make_canvas_tf(bounds, target_w):
    minx, miny, maxx, maxy = bounds
    scale = target_w / (maxx - minx)
    W = int(round((maxx - minx) * scale))
    H = int(round((maxy - miny) * scale))

    def to_canvas(xy):
        cx = (xy[:, 0] - minx) * scale
        cy = (maxy - xy[:, 1]) * scale
        return np.c_[cx, cy]

    return to_canvas, W, H


def atlases_bgra(atlases: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """PIL atlas pages -> OpenCV BGRA arrays for rasterisation."""
    out = {}
    for key, page in atlases.items():
        rgba = np.array(page["image"].convert("RGBA"))
        out[key] = np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
    return out


def prepare_raster_cache(parts, atlas_bgra):
    """Pre-crop atlas pages per part; mark rigid parts for single-warp path."""
    cache = []
    for p in parts:
        atlas = atlas_bgra[p["atlas"]]
        ah, aw = atlas.shape[:2]
        src = p["src_px"]
        pad = 2
        x0 = max(0, int(np.floor(src[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(src[:, 1].min())) - pad)
        x1 = min(aw, int(np.ceil(src[:, 0].max())) + pad)
        y1 = min(ah, int(np.ceil(src[:, 1].max())) + pad)
        src_local = src - np.array([x0, y0], dtype=np.float64)
        rigid = p["kind"] == "sprite" or (p["kind"] == "mesh" and p["single"])
        n = len(src_local)
        aff_idx = (0, max(1, n // 2), max(2, n - 1)) if n >= 3 else (0, 0, 0)
        cache.append(dict(
            part=p,
            atlas_crop=np.ascontiguousarray(atlas[y0:y1, x0:x1]),
            src_local=src_local,
            faces=p["faces"],
            rigid=rigid,
            aff_idx=aff_idx,
        ))
    return cache


def _composite_layer(canvas, layer, px0, py0):
    la = layer[:, :, 3:4].astype(np.float32) / 255.0
    lc = layer[:, :, :3].astype(np.float32)
    cv_roi = canvas[py0:py0 + layer.shape[0], px0:px0 + layer.shape[1]]
    ca = cv_roi[:, :, 3:4]
    out_a = la + ca * (1 - la)
    out_rgb = lc * la + cv_roi[:, :, :3] * ca * (1 - la)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_rgb = np.where(out_a > 0, out_rgb / np.maximum(out_a, 1e-6), 0)
    cv_roi[:, :, :3] = out_rgb
    cv_roi[:, :, 3:4] = out_a


def _rasterize_rigid(entry, cpos, px0, py0, pw, ph):
    crop = entry["atlas_crop"]
    src_local = entry["src_local"]
    layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    i0, i1, i2 = entry["aff_idx"]
    s = src_local[[i0, i1, i2]].astype(np.float32)
    d = (cpos[[i0, i1, i2]] - np.array([px0, py0], np.float32)).astype(np.float32)
    Maff = cv2.getAffineTransform(s, d)
    warped = cv2.warpAffine(
        crop, Maff, (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    mask = np.zeros((ph, pw), np.uint8)
    hull = cv2.convexHull(np.round(cpos - np.array([px0, py0])).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    layer[mask > 0] = warped[mask > 0]
    return layer


def _rasterize_skinned(entry, cpos, px0, py0, pw, ph):
    crop = entry["atlas_crop"]
    src_local = entry["src_local"]
    faces = entry["faces"]
    origin = np.array([px0, py0], np.float32)
    layer = np.zeros((ph, pw, 4), dtype=np.uint8)
    for f in faces:
        d = cpos[f].astype(np.float32)
        s = src_local[f].astype(np.float32)
        x0 = int(np.floor(d[:, 0].min()))
        x1 = int(np.ceil(d[:, 0].max()))
        y0 = int(np.floor(d[:, 1].min()))
        y1 = int(np.ceil(d[:, 1].max()))
        x0c, y0c = max(x0, px0), max(y0, py0)
        x1c, y1c = min(x1, px0 + pw), min(y1, py0 + ph)
        if x1c <= x0c or y1c <= y0c:
            continue
        dw, dh = x1c - x0c, y1c - y0c
        d_local = (d - origin).astype(np.float32)
        d_patch = d_local - np.array([x0c - px0, y0c - py0], np.float32)

        sx0 = max(0, int(np.floor(s[:, 0].min())))
        sy0 = max(0, int(np.floor(s[:, 1].min())))
        sx1 = min(crop.shape[1], int(np.ceil(s[:, 0].max())) + 1)
        sy1 = min(crop.shape[0], int(np.ceil(s[:, 1].max())) + 1)
        tri_crop = crop[sy0:sy1, sx0:sx1]
        if tri_crop.size == 0:
            continue
        s_rel = s - np.array([sx0, sy0], np.float32)

        Maff = cv2.getAffineTransform(
            np.ascontiguousarray(s_rel[:3], np.float32),
            np.ascontiguousarray(d_patch[:3], np.float32),
        )
        patch = cv2.warpAffine(
            tri_crop, Maff, (dw, dh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        mask = np.zeros((dh, dw), np.uint8)
        cv2.fillConvexPoly(
            mask, np.round(d_local - np.array([x0c - px0, y0c - py0])).astype(np.int32),
            255,
        )
        sel = mask > 0
        roi = layer[y0c - py0:y1c - py0, x0c - px0:x1c - px0]
        roi[sel] = patch[sel]
    return layer


def _render_part(entry, pos, to_canvas, W, H):
    cpos = to_canvas(pos)
    faces = entry["faces"]
    if len(faces) == 0:
        return None
    fv = cpos[faces.ravel()]
    px0 = max(int(np.floor(fv[:, 0].min())), 0)
    py0 = max(int(np.floor(fv[:, 1].min())), 0)
    px1 = min(int(np.ceil(fv[:, 0].max())), W)
    py1 = min(int(np.ceil(fv[:, 1].max())), H)
    if px1 <= px0 or py1 <= py0:
        return None
    pw, ph = px1 - px0, py1 - py0
    if entry["rigid"]:
        layer = _rasterize_rigid(entry, cpos, px0, py0, pw, ph)
    else:
        layer = _rasterize_skinned(entry, cpos, px0, py0, pw, ph)
    return px0, py0, layer


def rasterize(cache, positions, to_canvas, W, H, workers=0, opacities=None):
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    jobs = [
        (entry, pos, (1.0 if opacities is None else opacities[i]))
        for i, (entry, pos) in enumerate(zip(cache, positions))
    ]

    def run(entry, pos, opacity):
        if opacity < VIS_EPS:
            return None
        item = _render_part(entry, pos, to_canvas, W, H)
        if item is None:
            return None
        px0, py0, layer = item
        if opacity < 1.0 - VIS_EPS:
            layer = layer.copy()
            layer[:, :, 3] = np.clip(
                layer[:, :, 3].astype(np.float32) * opacity, 0, 255,
            ).astype(np.uint8)
        return px0, py0, layer

    n_workers = workers
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)

    if n_workers > 1 and len(jobs) > 8:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            layers = pool.map(lambda jp: run(*jp), jobs)
    else:
        layers = (run(entry, pos, opacity) for entry, pos, opacity in jobs)

    for item in layers:
        if item is None:
            continue
        px0, py0, layer = item
        _composite_layer(canvas, layer, px0, py0)

    out = np.zeros((H, W, 4), np.uint8)
    out[:, :, :3] = np.clip(canvas[:, :, :3], 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(canvas[:, :, 3] * 255, 0, 255).astype(np.uint8)
    return out


def export_gif(scene, clip, out_path, raster_cache, target_w=GIF_W, fps=GIF_FPS,
               bg=GIF_BG, workers=GIF_WORKERS):
    parts = scene["parts"]
    rest_TR = scene["TR"]
    hash2tr = scene["hash2tr"]

    sampler, stop, _ = decode_clip(clip)
    n_frames = max(1, int(round(stop * fps)))
    times = [i / fps for i in range(n_frames)]

    gmin = np.array([np.inf, np.inf])
    gmax = np.array([-np.inf, -np.inf])
    for t in times:
        ov = clip_overrides(clip, hash2tr, sampler, t)
        world = make_world(rest_TR, ov)
        positions = skin_all(parts, world)
        allxy = np.vstack(positions)
        gmin = np.minimum(gmin, allxy.min(0))
        gmax = np.maximum(gmax, allxy.max(0))

    w = gmax[0] - gmin[0]
    h = gmax[1] - gmin[1]
    bounds = (
        gmin[0] - w * PAD, gmin[1] - h * PAD,
        gmax[0] + w * PAD, gmax[1] + h * PAD,
    )
    to_canvas, W, H = make_canvas_tf(bounds, target_w)
    print(f"  {out_path.name}: {W}x{H}, {n_frames} frames @ {fps}fps")

    bg_rgb = np.array(
        [int(bg[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(round(1000 / fps))
    pal_img = None
    pil_frames = []

    for k, t in enumerate(times):
        ov = clip_overrides(clip, hash2tr, sampler, t)
        go_active, smr_alpha, smr_props = sample_clip_properties(clip, sampler, t)
        opacities = part_opacities(scene, go_active, smr_alpha, smr_props)
        world = make_world(rest_TR, ov)
        positions = skin_all(parts, world)
        out = rasterize(
            raster_cache, positions, to_canvas, W, H,
            workers=workers, opacities=opacities,
        )
        rgb = out[:, :, [2, 1, 0]].astype(np.float32)
        a = out[:, :, 3:4].astype(np.float32) / 255.0
        comp = np.clip(rgb * a + bg_rgb * (1 - a), 0, 255).astype(np.uint8)
        img = Image.fromarray(comp, "RGB")
        if pal_img is None:
            pal_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            pframe = pal_img
        else:
            pframe = img.quantize(palette=pal_img, dither=Image.Dither.NONE)
        pil_frames.append(pframe)
        print(f"    frame {k + 1}/{n_frames}", end="\r", flush=True)
    print()

    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0, optimize=True, disposal=1,
    )
    print(f"  wrote {out_path}")


def export_gifs(scene, gif_dir: Path, clip_filter: str | None = None,
                target_w=GIF_W, fps=GIF_FPS, bg=GIF_BG, workers=GIF_WORKERS):
    atlas_bgra = atlases_bgra(scene["atlases"])
    raster_cache = prepare_raster_cache(scene["parts"], atlas_bgra)
    rigid = sum(1 for e in raster_cache if e["rigid"])
    print(f"raster cache: {len(raster_cache)} parts ({rigid} rigid, "
          f"{len(raster_cache) - rigid} skinned)")
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    if clip_filter:
        wanted = [s.strip().lower() for s in clip_filter.split(",") if s.strip()]
        clips = [c for c in clips if any(w in c.m_Name.lower() for w in wanted)]
    print(f"exporting {len(clips)} gif(s) -> {gif_dir}")
    for clip in clips:
        out_path = gif_dir / f"anim_{clip.m_Name}.gif"
        try:
            export_gif(scene, clip, out_path, raster_cache, target_w, fps, bg, workers)
        except Exception as e:
            print(f"  !! failed {clip.m_Name}: {e}")


# ---------------------------------------------------------------------------
# AnimationClip decoding
# ---------------------------------------------------------------------------
def curve_size(binding):
    if binding.typeID == 4:
        if binding.attribute == 2:
            return 4
        return 3
    return 1


def decode_clip(clip):
    mc = clip.m_MuscleClip
    c = mc.m_Clip.data
    sc, dc, cc = c.m_StreamedClip, c.m_DenseClip, c.m_ConstantClip

    n_stream = int(sc.curveCount)
    n_dense = int(dc.m_CurveCount)
    const = np.array(cc.data, dtype=np.float64) if cc.data else np.zeros(0)
    n_total = n_stream + n_dense + len(const)

    per = {}
    if sc.data:
        buf = struct.pack("<%dI" % len(sc.data), *sc.data)
        f = np.frombuffer(buf, dtype="<f4")
        u = np.frombuffer(buf, dtype="<u4")
        ii = np.frombuffer(buf, dtype="<i4")
        pos, n = 0, len(u)
        while pos < n:
            t = float(f[pos])
            nk = int(u[pos + 1])
            pos += 2
            finite = -1e30 < t < 1e30
            for _ in range(nk):
                idx = int(ii[pos])
                coeff = f[pos + 1:pos + 5].astype(np.float64)
                pos += 5
                if finite:
                    per.setdefault(idx, []).append((t, coeff))
    stream_curves = {}
    for idx, keys in per.items():
        keys.sort(key=lambda x: x[0])
        ts = np.array([k[0] for k in keys])
        cf = np.array([k[1] for k in keys])
        stream_curves[idx] = (ts, cf)

    dense = np.array(dc.m_SampleArray, dtype=np.float64) if dc.m_SampleArray else np.zeros(0)
    dense_begin = float(dc.m_BeginTime)
    dense_rate = float(dc.m_SampleRate) or 1.0
    dense_frames = int(dc.m_FrameCount)

    def sampler(t):
        vals = np.zeros(n_total)
        for idx, (ts, cf) in stream_curves.items():
            j = bisect_right(ts, t) - 1
            if j < 0:
                j = 0
            kt = ts[j]
            co = cf[j]
            dt = t - kt
            vals[idx] = ((co[0] * dt + co[1]) * dt + co[2]) * dt + co[3]
        if n_dense > 0 and dense.size:
            fr = (t - dense_begin) * dense_rate
            f0 = int(np.clip(np.floor(fr), 0, dense_frames - 1))
            f1 = min(f0 + 1, dense_frames - 1)
            a = fr - f0
            row0 = dense[f0 * n_dense:(f0 + 1) * n_dense]
            row1 = dense[f1 * n_dense:(f1 + 1) * n_dense]
            vals[n_stream:n_stream + n_dense] = row0 * (1 - a) + row1 * a
        if len(const):
            vals[n_stream + n_dense:] = const
        return vals

    return sampler, float(mc.m_StopTime), n_total


def clip_overrides(clip, hash2tr, sampler, t):
    vals = sampler(t)
    gb = clip.m_ClipBindingConstant.genericBindings
    out = {}
    idx = 0
    for b in gb:
        size = curve_size(b)
        if b.typeID == 4:
            tr = hash2tr.get(b.path)
            if tr is not None:
                o = out.setdefault(tr, {})
                if b.attribute == 1:
                    o["pos"] = vals[idx:idx + 3]
                elif b.attribute == 2:
                    q = vals[idx:idx + 4]
                    nrm = np.linalg.norm(q)
                    o["rot"] = (q / nrm) if nrm > 1e-9 else q
                elif b.attribute == 3:
                    o["scale"] = vals[idx:idx + 3]
        idx += size
    return out


def path_hash(path: str) -> int:
    return zlib.crc32(path.encode("utf-8")) & 0xFFFFFFFF


def sample_clip_properties(clip, sampler, t) -> tuple[dict[int, float], dict[int, float], dict[int, dict[int, float]]]:
    """Sample GameObject active, SMR alpha, and SMR blend weights at time *t*."""
    vals = sampler(t)
    go_active: dict[int, float] = {}
    smr_alpha: dict[int, float] = {}
    smr_props: dict[int, dict[int, float]] = {}
    idx = 0
    for b in clip.m_ClipBindingConstant.genericBindings:
        size = curve_size(b)
        if b.typeID == GO_TYPE and b.attribute == GO_ACTIVE_ATTR:
            go_active[b.path] = float(vals[idx])
        elif b.typeID == SMR_TYPE:
            value = float(vals[idx])
            if b.attribute == SMR_COLOR_A_ATTR:
                smr_alpha[b.path] = value
            else:
                smr_props.setdefault(b.path, {})[b.attribute] = value
        idx += size
    return go_active, smr_alpha, smr_props


def _mouth_bone_active(scene, go_active, bone_name: str) -> bool:
    TR = scene["TR"]
    tr_to_path = scene["tr_to_path"]
    go2tr = scene["go2tr"]
    go_name = scene["go_name"]
    tr2go = {t: g for g, t in go2tr.items()}
    for tr, go in tr2go.items():
        if go_name.get(go) == bone_name:
            return transform_active(tr, go_active, tr_to_path, TR, scene.get("tr_active_static"))
    return True


def _mouth_crossfade_opacity(part_name, scene, go_active, smr_props) -> float | None:
    """Cross-fade between Mouth_A and Mouth_Idle when both bones are enabled."""
    if part_name not in (MOUTH_A_PART, MOUTH_IDLE_PART):
        return None
    a_on = _mouth_bone_active(scene, go_active, "Bone_Mouth_A")
    idle_on = _mouth_bone_active(scene, go_active, "Bone_Mouth_Idle")
    if not a_on and not idle_on:
        return 0.0
    if part_name == MOUTH_A_PART:
        if not a_on:
            return 0.0
        if not idle_on:
            return 1.0
    else:
        if not idle_on:
            return 0.0
        if not a_on:
            return 1.0

    idle_part = next(p for p in scene["parts"] if p["name"] == MOUTH_IDLE_PART)
    idle_path = scene["tr_to_path"].get(idle_part["go_tr_pid"])
    if idle_path is None:
        return 1.0
    props = smr_props.get(path_hash(idle_path), {})
    idle_weight = props.get(SMR_BLEND_CROSSFADE, 100.0) / 100.0
    idle_weight = max(0.0, min(1.0, idle_weight))
    if part_name == MOUTH_IDLE_PART:
        return idle_weight
    return 1.0 - idle_weight


def transform_active(tr, go_active, tr_to_path, TR, tr_active_static=None) -> bool:
    """False when this transform or an ancestor is disabled - either by this
    clip's own active-toggle curve, or (when the clip never touches it) by
    the object's static default state (many alt-expression parts start
    disabled and are only switched on by the clip that uses them)."""
    tr_active_static = tr_active_static or {}
    while tr:
        path = tr_to_path.get(tr)
        active = None
        if path is not None:
            active = go_active.get(path_hash(path))
        if active is None:
            active = 1.0 if tr_active_static.get(tr, True) else 0.0
        if active < 0.5:
            return False
        tr = TR.get(tr, {}).get("father", 0)
    return True


def _smr_blend_shape_weight(smr_props: dict[int, dict[int, float]], path_h: int) -> float | None:
    """Optional SMR blend / shader weight keyed by mesh path hash."""
    weight = smr_props.get(path_h, {}).get(SMR_BLEND_SHAPE_WEIGHT)
    if weight is None:
        return None
    return max(0.0, min(1.0, weight))


def part_opacities(scene, go_active, smr_alpha, smr_props) -> list[float]:
    """Per-part visibility/alpha from GameObject active + SMR curves."""
    TR = scene["TR"]
    tr_to_path = scene["tr_to_path"]
    out: list[float] = []
    for part in scene["parts"]:
        mouth_op = _mouth_crossfade_opacity(part["name"], scene, go_active, smr_props)
        if mouth_op is not None:
            out.append(mouth_op)
            continue

        go_tr = part.get("go_tr_pid") or part.get("tr_pid")
        path_h = None
        if go_tr:
            path = tr_to_path.get(go_tr)
            if path is not None:
                path_h = path_hash(path)

        blend = _smr_blend_shape_weight(smr_props, path_h) if path_h is not None else None
        if part["name"] in BALGURE_PARTS and blend is not None:
            opacity = blend
            if path_h is not None:
                alpha = smr_alpha.get(path_h)
                if alpha is not None:
                    opacity *= max(0.0, min(1.0, alpha))
            out.append(opacity)
            continue

        if go_tr and not transform_active(go_tr, go_active, tr_to_path, TR, scene.get("tr_active_static")):
            out.append(0.0)
            continue
        opacity = 1.0
        if path_h is not None:
            alpha = smr_alpha.get(path_h)
            if alpha is not None:
                opacity *= max(0.0, min(1.0, alpha))
        alpha = smr_alpha.get(part.get("name_hash", 0))
        if alpha is not None:
            opacity *= max(0.0, min(1.0, alpha))
        if blend is not None:
            opacity *= blend
        out.append(opacity)
    return out


# ---------------------------------------------------------------------------
# Unity bundle -> scene dict
# ---------------------------------------------------------------------------
def load_atlases(objs) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Return atlases keyed by partN and Texture2D path_id -> atlas key."""
    atlases: dict[str, dict[str, Any]] = {}
    tex2atlas: dict[int, str] = {}
    unmatched: list[tuple[int, Any]] = []
    for o in objs:
        if o.type.name != "Texture2D":
            continue
        data = o.read()
        key = atlas_key_from_name(data.m_Name)
        if key is None:
            unmatched.append((o.path_id, data))
            continue
        img = data.image
        w, h = img.size
        atlases[key] = {
            "key": key,
            "name": data.m_Name or key,
            "image": img,
            "width": w,
            "height": h,
        }
        tex2atlas[o.path_id] = key

    # No partN/partsN-suffixed texture found. If there's exactly one
    # candidate, it's the whole atlas (e.g. "FullbodyIMG_X_N"). If there are
    # several (a body atlas alongside small unrelated utility/gizmo/UI
    # textures - IK handles, bone dots, glow FX...), assume the largest one
    # by pixel area is the real atlas page rather than failing outright.
    if not atlases and unmatched:
        path_id, data = max(unmatched, key=lambda pair: pair[1].image.size[0] * pair[1].image.size[1])
        img = data.image
        w, h = img.size
        key = "part1"
        atlases[key] = {
            "key": key,
            "name": data.m_Name or key,
            "image": img,
            "width": w,
            "height": h,
        }
        tex2atlas[path_id] = key

    return atlases, tex2atlas


def build_scene_from_objs(objs) -> dict[str, Any]:
    byid = {o.path_id: o for o in objs}

    TR = {}
    go2tr = {}
    go_name = {}
    go_active_static: dict[int, bool] = {}
    tr_children = {}
    for o in objs:
        if o.type.name == "Transform":
            d = o.read()
            p, r, s = d.m_LocalPosition, d.m_LocalRotation, d.m_LocalScale
            TR[o.path_id] = dict(
                pos=(p.x, p.y, p.z),
                rot=(r.x, r.y, r.z, r.w),
                scale=(s.x, s.y, s.z),
                father=getattr(d.m_Father, "path_id", 0),
            )
            go2tr[getattr(d.m_GameObject, "path_id", 0)] = o.path_id
            tr_children[o.path_id] = [getattr(c, "path_id", 0) for c in d.m_Children]
        elif o.type.name == "GameObject":
            gd = o.read()
            go_name[o.path_id] = gd.m_Name
            go_active_static[o.path_id] = bool(getattr(gd, "m_IsActive", True))

    # Many parts (alternate face expressions etc.) start disabled and are
    # only switched on by specific clips; index that baseline by transform
    # so a clip that never touches a given object doesn't wrongly show it.
    tr_active_static = {
        tr: go_active_static.get(gid, True) for gid, tr in go2tr.items()
    }

    atlases, tex2atlas = load_atlases(objs)
    if not atlases:
        raise RuntimeError("no atlas textures found (expected names ending with part1, part2, …)")

    def atlas_for_material(mat_pid):
        mo = byid.get(mat_pid)
        if not mo:
            return None
        key = atlas_key_from_name(mo.read().m_Name)
        if key is None and len(atlases) == 1:
            key = next(iter(atlases))
        return key

    parts = []
    for o in objs:
        if o.type.name != "SkinnedMeshRenderer":
            continue
        smr = o.read()
        mp = getattr(smr.m_Mesh, "path_id", 0)
        if mp not in byid:
            continue
        mesh = byid[mp].read()
        mats = smr.m_Materials or []
        atlas = atlas_for_material(getattr(mats[0], "path_id", 0)) if mats else None
        if atlas is None or atlas not in atlases:
            # Material names a partN/partsN page with no matching texture
            # anywhere in the bundle - an orphaned/unused material, not
            # something we can render.
            continue

        bones = [getattr(b, "path_id", 0) for b in smr.m_Bones]
        bind = [mat4(m) for m in mesh.m_BindPose]

        h = MeshHandler(mesh)
        h.process()
        vc = h.m_VertexCount
        v = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)
        uv = np.array(h.m_UV0, dtype=np.float64).reshape(vc, 2)
        vh = np.c_[v, np.ones(vc)]

        single = len(bones) <= 1
        bi = bw = None
        if not single:
            bi_raw = np.array(h.m_BoneIndices)
            # Guard: UnityPy may return a single object instead of an empty array
            # for meshes with no bone data (e.g. parts_Head with bones=0).
            if bi_raw.dtype == object or bi_raw.size < vc:
                single = True
            else:
                k = max(1, bi_raw.size // vc)
                bi = bi_raw.reshape(vc, k)
                if h.m_BoneWeights is not None and np.array(h.m_BoneWeights).size == vc * k:
                    bw = np.array(h.m_BoneWeights, dtype=np.float64).reshape(vc, k)
                else:
                    bw = np.zeros((vc, k))
                    bw[:, 0] = 1.0

        faces = []
        for sm in h.get_triangles():
            faces.append(np.array(sm).reshape(-1, 3))
        faces = np.vstack(faces) if faces else np.zeros((0, 3), int)

        aw = atlases[atlas]["width"]
        ah = atlases[atlas]["height"]
        src_px = np.c_[uv[:, 0] * aw, (1 - uv[:, 1]) * ah]
        go_tr_pid = go2tr.get(getattr(smr.m_GameObject, "path_id", 0))
        parts.append(dict(
            name=mesh.m_Name, atlas=atlas, kind="mesh",
            bones=bones, bind=bind, vh=vh, single=single,
            bi=bi, bw=bw, src_px=src_px, faces=faces,
            order=getattr(smr, "m_SortingOrder", 0),
            go_tr_pid=go_tr_pid,
            name_hash=path_hash(mesh.m_Name),
        ))

    for o in objs:
        if o.type.name != "SpriteRenderer":
            continue
        sr = o.read()
        sp_ptr = getattr(sr, "m_Sprite", None)
        if not sp_ptr or getattr(sp_ptr, "path_id", 0) not in byid:
            continue
        sp_o = byid[sp_ptr.path_id]
        sp = sp_o.read()
        rd = sp.m_RD
        tex_pid = getattr(getattr(rd, "texture", None), "path_id", 0)
        atlas = tex2atlas.get(tex_pid)
        if atlas is None:
            continue
        tr_pid = go2tr.get(getattr(sr.m_GameObject, "path_id", 0))
        if tr_pid is None:
            continue

        rect = sp.m_Rect
        ptu = sp.m_PixelsToUnits or 100.0
        piv = sp.m_Pivot

        h = MeshHandler(rd, sp_o.version)
        h.process()
        vc = h.m_VertexCount
        sv = np.array(h.m_Vertices, dtype=np.float64).reshape(vc, 3)

        # UV lookup uses the un-flipped mesh - flip mirrors where the quad
        # sits in local/world space, not which source pixels it samples.
        ah = atlases[atlas]["height"]
        ax = rect.x + (sv[:, 0] * ptu + piv.x * rect.width)
        ay = rect.y + (sv[:, 1] * ptu + piv.y * rect.height)
        src = np.c_[ax, ah - ay]

        sv_local = sv.copy()
        if getattr(sr, "m_FlipX", False):
            sv_local[:, 0] = -sv_local[:, 0]
        if getattr(sr, "m_FlipY", False):
            sv_local[:, 1] = -sv_local[:, 1]
        sv2 = np.c_[sv_local[:, :2], np.zeros(vc), np.ones(vc)]

        faces = []
        for sm in h.get_triangles():
            faces.append(np.array(sm).reshape(-1, 3))
        faces = np.vstack(faces) if faces else np.zeros((0, 3), int)

        parts.append(dict(
            name=sp.m_Name, atlas=atlas, kind="sprite",
            tr_pid=tr_pid, sv2=sv2, src_px=src, faces=faces,
            order=getattr(sr, "m_SortingOrder", 0),
            go_tr_pid=tr_pid,
            name_hash=path_hash(sp.m_Name),
        ))

    parts.sort(key=lambda p: p["order"])

    animator_go = None
    for o in objs:
        if o.type.name == "Animator":
            animator_go = getattr(o.read().m_GameObject, "path_id", 0)
            break
    hash2tr = {}
    tr_to_path = {}
    if animator_go is not None and animator_go in go2tr:
        root_tr = go2tr[animator_go]
        tr2go = {t: g for g, t in go2tr.items()}

        def name_of(tr):
            return go_name.get(tr2go.get(tr, 0), "")

        def walk(tr, prefix):
            for c in tr_children.get(tr, []):
                nm = name_of(c)
                p = nm if prefix == "" else prefix + "/" + nm
                tr_to_path[c] = p
                hash2tr[path_hash(p)] = c
                walk(c, p)

        walk(root_tr, "")
        tr_to_path[root_tr] = ""
        hash2tr[path_hash("")] = root_tr

    return dict(
        objs=objs, TR=TR, atlases=atlases, parts=parts, hash2tr=hash2tr,
        tr_to_path=tr_to_path, go2tr=go2tr, go_name=go_name,
        tr_active_static=tr_active_static,
    )


# ---------------------------------------------------------------------------
# Multi-variant bundles: some files pack several skins (N, NS1, NS2, …) of
# the same character - each with its own Animator-rooted bone hierarchy and
# its own (same-named) set of AnimationClips - into a single AssetBundle.
# The functions below detect those sub-skeletons and split the bundle's
# objects into one self-contained group per variant, so each is exported as
# its own clean Spine skeleton instead of being merged into one.
# ---------------------------------------------------------------------------
SCOPED_TYPES = {
    "Transform", "GameObject", "SkinnedMeshRenderer", "SpriteRenderer",
    "Animator", "AnimationClip", "Texture2D",
}


def build_hash2tr(root_tr, tr_children, go2tr, go_name):
    tr2go = {t: g for g, t in go2tr.items()}

    def name_of(tr):
        return go_name.get(tr2go.get(tr, 0), "")

    hash2tr = {}
    tr_to_path = {}

    def walk(tr, prefix):
        for c in tr_children.get(tr, []):
            nm = name_of(c)
            p = nm if prefix == "" else prefix + "/" + nm
            tr_to_path[c] = p
            hash2tr[path_hash(p)] = c
            walk(c, p)

    walk(root_tr, "")
    tr_to_path[root_tr] = ""
    hash2tr[path_hash("")] = root_tr
    return hash2tr, tr_to_path


def discover_char_roots(objs):
    """Find distinct rigged sub-skeletons: Animator roots (with an actual
    AnimatorController - excludes stray unused reference meshes that also
    happen to carry an Animator component) whose subtree contains a
    SkinnedMeshRenderer, keeping only the outermost ones."""
    go2tr = {}
    go_name = {}
    tr_children = {}
    for o in objs:
        if o.type.name == "Transform":
            d = o.read()
            go2tr[getattr(d.m_GameObject, "path_id", 0)] = o.path_id
            tr_children[o.path_id] = [getattr(c, "path_id", 0) for c in d.m_Children]
        elif o.type.name == "GameObject":
            go_name[o.path_id] = o.read().m_Name
    tr2go = {t: g for g, t in go2tr.items()}

    smr_trs = set()
    animator_roots = []  # (tr, controller_path_id)
    for o in objs:
        if o.type.name in ("SkinnedMeshRenderer", "SpriteRenderer"):
            tr = go2tr.get(getattr(o.read().m_GameObject, "path_id", 0))
            if tr is not None:
                smr_trs.add(tr)
        elif o.type.name == "Animator":
            d = o.read()
            tr = go2tr.get(getattr(d.m_GameObject, "path_id", 0))
            ctrl_pid = getattr(d.m_Controller, "path_id", 0)
            if tr is not None and ctrl_pid:
                animator_roots.append((tr, ctrl_pid))

    def subtree(root):
        seen = set()
        stack = [root]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(tr_children.get(t, []))
        return seen

    candidates = [(r, ctrl, subtree(r)) for r, ctrl in animator_roots]
    candidates = [(r, ctrl, sub) for r, ctrl, sub in candidates if sub & smr_trs]
    maximal = [
        (root, ctrl, sub) for root, ctrl, sub in candidates
        if not any(root != other and root in other_sub for other, _, other_sub in candidates)
    ]

    controller_names = {}
    for o in objs:
        if o.type.name == "AnimatorController":
            try:
                controller_names[o.path_id] = o.read().m_Name
            except Exception:
                pass

    used_names: dict[str, int] = {}
    roots_info = []
    seen_roots = set()
    for root, ctrl, sub in maximal:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        raw = (
            controller_names.get(ctrl)
            or go_name.get(tr2go.get(root, 0))
            or f"root_{root}"
        )
        raw = re.sub(r"_?[Cc]on(troller)?$", "", raw)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or f"root_{root}"
        if name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 0
        roots_info.append((root, sub, name, ctrl))

    return roots_info, go2tr, go_name, tr_children


def collect_variant_texture_names(objs, subtree, go2tr):
    """Names of textures actually used by this variant's parts: direct
    Sprite->texture references, plus the material names its
    SkinnedMeshRenderers use (textures/materials share names by convention).
    Also returns the partN/partsN keys those materials resolve to, so a
    globally-unique key can be scoped to only the variant that wants it."""
    byid = {o.path_id: o for o in objs}
    names = set()
    keys = set()
    for o in objs:
        if o.type.name not in ("SkinnedMeshRenderer", "SpriteRenderer"):
            continue
        d = o.read()
        tr = go2tr.get(getattr(d.m_GameObject, "path_id", 0))
        if tr is None or tr not in subtree:
            continue
        if o.type.name == "SkinnedMeshRenderer":
            for m in d.m_Materials or []:
                mo = byid.get(getattr(m, "path_id", 0))
                if mo:
                    try:
                        mname = mo.read().m_Name
                        names.add(mname)
                        key = atlas_key_from_name(mname)
                        if key is not None:
                            keys.add(key)
                    except Exception:
                        pass
        else:
            sp_ptr = getattr(d, "m_Sprite", None)
            sp_o = byid.get(getattr(sp_ptr, "path_id", 0)) if sp_ptr else None
            if sp_o:
                try:
                    tex_ptr = getattr(sp_o.read().m_RD, "texture", None)
                    tex_o = byid.get(getattr(tex_ptr, "path_id", 0))
                    if tex_o:
                        names.add(tex_o.read().m_Name)
                except Exception:
                    pass
    return names, keys


def collect_ambiguous_atlas_keys(objs):
    """partN/partsN keys shared by more than one Texture2D in the whole
    bundle - only these actually need per-variant disambiguation; a unique
    key can never collide across variants, so always keep it."""
    groups: dict[str, int] = {}
    for o in objs:
        if o.type.name != "Texture2D":
            continue
        try:
            key = atlas_key_from_name(o.read().m_Name)
        except Exception:
            key = None
        if key is not None:
            groups[key] = groups.get(key, 0) + 1
    return {k for k, n in groups.items() if n > 1}


def build_objs_for_variant(objs, subtree, assigned_clip_pids, root_tr, go2tr, texture_names, used_keys, ambiguous_keys):
    out = []
    for o in objs:
        t = o.type.name
        if t not in SCOPED_TYPES:
            out.append(o)
            continue
        if t == "Transform":
            if o.path_id in subtree:
                out.append(o)
        elif t == "GameObject":
            tr = go2tr.get(o.path_id)
            if tr is not None and tr in subtree:
                out.append(o)
        elif t in ("SkinnedMeshRenderer", "SpriteRenderer"):
            tr = go2tr.get(getattr(o.read().m_GameObject, "path_id", 0))
            if tr is not None and tr in subtree:
                out.append(o)
        elif t == "Animator":
            if go2tr.get(getattr(o.read().m_GameObject, "path_id", 0)) == root_tr:
                out.append(o)
        elif t == "AnimationClip":
            if o.path_id in assigned_clip_pids:
                out.append(o)
        elif t == "Texture2D":
            try:
                tname = o.read().m_Name
            except Exception:
                tname = None
            key = atlas_key_from_name(tname) if tname else None
            if key is not None and key not in ambiguous_keys and key in used_keys:
                out.append(o)  # unique partN/partsN key this variant's own materials want
            elif not texture_names:
                out.append(o)  # fallback: no resolvable name -> keep all
            elif tname in texture_names:
                out.append(o)
    return out


def split_scene(objs):
    """[(variant_name_or_None, objs), ...]. A single-skeleton bundle comes
    back unchanged as one [(None, objs)] entry."""
    roots_info, go2tr, go_name, tr_children = discover_char_roots(objs)
    if len(roots_info) < 2:
        return [(None, objs)]

    all_clip_pids = {o.path_id for o in objs if o.type.name == "AnimationClip"}

    # Ground truth: each variant's own AnimatorController lists exactly the
    # clips it uses (variants can share identical bone-name conventions, so
    # clip bindings alone can't tell two skeletons apart - see PR notes).
    controller_clips: dict[int, set[int]] = {}
    for o in objs:
        if o.type.name != "AnimatorController":
            continue
        d = o.read()
        pids = {getattr(p, "path_id", 0) for p in (d.m_AnimationClips or [])}
        controller_clips[o.path_id] = pids & all_clip_pids

    assigned: dict[int, set[int]] = {root_tr: set() for root_tr, _, _, _ in roots_info}
    covered: set[int] = set()
    for root_tr, _, _, ctrl in roots_info:
        clips = controller_clips.get(ctrl)
        if clips:
            assigned[root_tr] |= clips
            covered |= clips

    # Fallback for any clip no controller claimed: best bone-path match.
    leftover = all_clip_pids - covered
    if leftover:
        root_hash = {
            root_tr: build_hash2tr(root_tr, tr_children, go2tr, go_name)[0]
            for root_tr, _, _, _ in roots_info
        }
        byid = {o.path_id: o for o in objs}
        for pid in leftover:
            clip = byid[pid].read()
            gb = clip.m_ClipBindingConstant.genericBindings
            best_root, best_score = None, 0
            for root_tr, _, _, _ in roots_info:
                score = sum(1 for b in gb if b.typeID == 4 and b.path in root_hash[root_tr])
                if score > best_score:
                    best_score, best_root = score, root_tr
            if best_root is not None:
                assigned[best_root].add(pid)

    ambiguous_keys = collect_ambiguous_atlas_keys(objs)
    out = []
    for root_tr, sub, name, _ in roots_info:
        tex_names, used_keys = collect_variant_texture_names(objs, sub, go2tr)
        variant_objs = build_objs_for_variant(
            objs, sub, assigned[root_tr], root_tr, go2tr, tex_names, used_keys, ambiguous_keys,
        )
        out.append((name, variant_objs))
    return out


def load_scene(src: Path) -> list[tuple[str | None, dict[str, Any]]]:
    env = UnityPy.load(str(src))
    objs = list(env.objects)
    scenes = []
    for name, variant_objs in split_scene(objs):
        try:
            scenes.append((name, build_scene_from_objs(variant_objs)))
        except RuntimeError as e:
            label = name or "(default)"
            print(f"  !! variant {label} skipped: {e}")
    if not scenes:
        if not any(o.type.name == "Texture2D" for o in objs):
            raise RuntimeError(
                "no exportable variant found in this bundle: it has no "
                "Texture2D objects at all - its atlas is stored in a "
                "different AssetBundle this file depends on, which this "
                "tool can't follow"
            )
        raise RuntimeError("no exportable variant found in this bundle")
    return scenes


# ---------------------------------------------------------------------------
# Spine skeleton
# ---------------------------------------------------------------------------
def build_bones(scene):
    TR = scene["TR"]
    go2tr = scene["go2tr"]
    go_name = scene["go_name"]
    tr2go = {t: g for g, t in go2tr.items()}

    def raw_name(tr):
        return go_name.get(tr2go.get(tr, 0)) or f"bone_{tr}"

    used = {}
    tr_name = {}
    for tr in TR:
        nm = raw_name(tr).replace("/", "_")
        if nm in used:
            used[nm] += 1
            nm = f"{nm}#{used[nm]}"
        else:
            used[nm] = 0
        tr_name[tr] = nm

    children = {tr: [] for tr in TR}
    roots = []
    for tr, t in TR.items():
        f = t["father"]
        if f and f in TR:
            children[f].append(tr)
        else:
            roots.append(tr)

    ordered = []
    parent_name = {}
    stack = list(reversed(roots))
    while stack:
        tr = stack.pop()
        ordered.append(tr)
        for c in reversed(children[tr]):
            parent_name[c] = tr
            stack.append(c)

    bones = [{"name": "root"}]
    setup = {}
    bone_index = {"root": 0}
    for tr in ordered:
        s = trs_to_spine(TR[tr])
        setup[tr] = s
        parent = tr_name[parent_name[tr]] if tr in parent_name else "root"
        b = {"name": tr_name[tr], "parent": parent}
        if abs(s["x"]) > EPS:
            b["x"] = r2(s["x"])
        if abs(s["y"]) > EPS:
            b["y"] = r2(s["y"])
        if abs(s["rotation"]) > EPS:
            b["rotation"] = r2(s["rotation"])
        if abs(s["scaleX"] - 1) > EPS:
            b["scaleX"] = r2(s["scaleX"])
        if abs(s["scaleY"] - 1) > EPS:
            b["scaleY"] = r2(s["scaleY"])
        if abs(s["shearY"]) > EPS:
            b["shearY"] = r2(s["shearY"])
        bones.append(b)
        bone_index[tr_name[tr]] = len(bones) - 1

    return bones, tr_name, bone_index, setup


def part_weighted_vertices(part, tr_name, bone_index):
    out = []
    if part["kind"] == "sprite":
        bidx = bone_index[tr_name[part["tr_pid"]]]
        for v in part["sv2"]:
            out += [1, bidx, r2(v[0]), r2(v[1]), 1.0]
        return out

    bones = part["bones"]
    bind = part["bind"]
    vh = part["vh"]
    bone_idx = [bone_index[tr_name[b]] for b in bones]

    if part["single"]:
        if len(bones) == 0:
            # No bones at all – treat as identity transform
            for p in vh:
                out += [1, -1, r2(p[0]), r2(p[1]), 1.0]
            return out
        local = (bind[0] @ vh.T).T
        for p in local:
            out += [1, bone_idx[0], r2(p[0]), r2(p[1]), 1.0]
        return out

    bi, bw = part["bi"], part["bw"]
    vc, k = bi.shape
    locals_by_bone = {j: (bind[j] @ vh.T).T for j in range(len(bones))}

    for i in range(vc):
        acc = {}
        for ki in range(k):
            w = float(bw[i, ki])
            if w <= 0:
                continue
            j = int(bi[i, ki])
            acc[j] = acc.get(j, 0.0) + w
        if not acc:
            acc[int(bi[i, 0])] = 1.0
        out.append(len(acc))
        for j, w in acc.items():
            p = locals_by_bone[j][i]
            out += [bone_idx[j], r2(p[0]), r2(p[1]), r2(w)]
    return out


def mesh_image_size(part, atlases):
    """Mesh attachment image size must match the exported atlas page PNG."""
    page = atlases[part["atlas"]]
    return int(page["width"]), int(page["height"])


def _mesh_edge_pairs(faces: np.ndarray) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _mesh_hull_order(faces: np.ndarray, vcount: int) -> tuple[int, list[int]]:
    """Return hull length and vertex order with boundary vertices first."""
    edge_use: dict[tuple[int, int], int] = {}
    for a, b, c in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (min(i, j), max(i, j))
            edge_use[key] = edge_use.get(key, 0) + 1

    boundary = [edge for edge, count in edge_use.items() if count == 1]
    if not boundary:
        return vcount, list(range(vcount))

    adj: dict[int, list[int]] = {}
    for i, j in boundary:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    start = boundary[0][0]
    hull = [start]
    prev = -1
    cur = start
    used: set[tuple[int, int]] = set()
    for _ in range(len(boundary) + 1):
        nxt = None
        for cand in adj.get(cur, []):
            if cand == prev:
                continue
            key = (min(cur, cand), max(cur, cand))
            if key in used:
                continue
            nxt = cand
            used.add(key)
            break
        if nxt is None or nxt == start:
            break
        hull.append(nxt)
        prev, cur = cur, nxt

    hull_set = {i for edge in boundary for i in edge}
    for v in sorted(hull_set - set(hull)):
        hull.append(v)

    internal = sorted(set(range(vcount)) - set(hull))
    return len(hull), hull + internal


def _reorder_part_vertices(part, order: list[int]):
    order_arr = np.asarray(order, dtype=np.int64)
    rp = dict(part)
    rp["src_px"] = part["src_px"][order_arr]
    if part["kind"] == "sprite":
        rp["sv2"] = part["sv2"][order_arr]
    else:
        rp["vh"] = part["vh"][order_arr]
        if not part["single"]:
            rp["bi"] = part["bi"][order_arr]
            rp["bw"] = part["bw"][order_arr]
    old_to_new = {old: new for new, old in enumerate(order)}
    rp["faces"] = np.vectorize(old_to_new.get)(part["faces"])
    return rp


def build_spine_mesh_attachment(part, tr_name, bone_index, aw, ah):
    """Build mesh attachment arrays with Spine hull order and edge pairs."""
    faces = np.asarray(part["faces"], dtype=np.int64)
    edge_pairs = _mesh_edge_pairs(faces)
    hull_len, order = _mesh_hull_order(faces, len(part["src_px"]))
    old_to_new = {old: new for new, old in enumerate(order)}
    rp = _reorder_part_vertices(part, order)

    uvs = []
    for sx, sy in rp["src_px"]:
        uvs += [r2(sx / aw), r2(sy / ah)]
    tris = [int(i) for i in rp["faces"].ravel()]
    verts = part_weighted_vertices(rp, tr_name, bone_index)
    edges = []
    for i, j in edge_pairs:
        edges.extend([old_to_new[i], old_to_new[j]])
    return hull_len, uvs, tris, verts, edges


def part_slot_names(parts) -> list[str]:
    """Spine slot names for scene parts (matches build_slots_and_skin naming)."""
    used_names: dict[str, int] = {}
    names: list[str] = []
    for p in parts:
        base = p["name"] or f"part_{p['order']}"
        name = base
        if name in used_names:
            used_names[name] += 1
            name = f"{name}#{used_names[name]}"
        else:
            used_names[name] = 0
        names.append(name)
    return names


def _opacity_key_stepped(prev: float | None, cur: float) -> bool:
    """Use stepped interpolation for hard on/off visibility switches."""
    if prev is None or abs(cur - prev) <= EPS:
        return False
    if abs(cur - prev) >= 0.5:
        return True
    near_edge = lambda x: x < VIS_EPS or x > 1.0 - VIS_EPS
    return near_edge(prev) and near_edge(cur)


def build_slot_opacity_tracks(scene, clip, sampler, times) -> dict:
    """Slot color (alpha) keyframes from GameObject active / SMR alpha curves."""
    part_names = part_slot_names(scene["parts"])
    series = {name: [] for name in part_names}
    for t in times:
        go_active, smr_alpha, smr_props = sample_clip_properties(clip, sampler, t)
        opacities = part_opacities(scene, go_active, smr_alpha, smr_props)
        for name, op in zip(part_names, opacities):
            series[name].append(max(0.0, min(1.0, op)))

    slots_out = {}
    for name, vals in series.items():
        if all(abs(v - 1.0) <= EPS for v in vals):
            continue

        n = len(vals)
        keep = [True] * n
        for i in range(1, n - 1):
            if vals[i] == vals[i - 1] and vals[i] == vals[i + 1]:
                keep[i] = False

        prev_v = None
        keys = []
        for i in range(n):
            if not keep[i]:
                continue
            v = vals[i]
            key = {
                "time": r2(times[i]),
                "color": f"ffffff{int(round(v * 255)):02x}",
            }
            if _opacity_key_stepped(prev_v, v):
                key["curve"] = "stepped"
            keys.append(key)
            prev_v = v

        if keys:
            slots_out[name] = {"color": keys}
    return slots_out


def build_slots_and_skin(scene, tr_name, bone_index, editor=False):
    parts = scene["parts"]
    atlases = scene["atlases"]
    TR = scene["TR"]
    tr_to_path = scene["tr_to_path"]
    tr_active_static = scene.get("tr_active_static") or {}

    slots = []
    attachments = {}
    for p, name in zip(parts, part_slot_names(parts)):
        page = atlases[p["atlas"]]
        aw, ah = page["width"], page["height"]
        try:
            hull_len, uvs, tris, verts, edges = build_spine_mesh_attachment(
                p, tr_name, bone_index, aw, ah,
            )
        except (KeyError, IndexError) as e:
            # Some meshes reference a bone slot Unity never actually bound
            # (a null bone pointer, or one outside this skeleton) - skip
            # just that part rather than failing the whole export.
            print(f"  !! part {name} skipped: {e!r}")
            continue
        slot = {"name": name, "bone": "root", "attachment": name}
        go_tr = p.get("go_tr_pid") or p.get("tr_pid")
        if go_tr and not transform_active(go_tr, {}, tr_to_path, TR, tr_active_static):
            # Starts disabled in Unity (e.g. an alt facial expression) -
            # hide it in the setup pose; clips that use it re-enable it.
            slot["color"] = "ffffff00"
        slots.append(slot)
        width, height = mesh_image_size(p, atlases)

        attachments[name] = {
            name: {
                "type": "mesh",
                "uvs": uvs,
                "triangles": tris,
                "vertices": verts,
                "hull": hull_len,
                "edges": edges,
                "width": width,
                "height": height,
                "path": page["name"],
            }
        }

    skins = [{"name": "default", "attachments": attachments}]
    return slots, skins


def reduce_keys(times, values, key_builder):
    n = len(values)
    keep = [True] * n
    for i in range(1, n - 1):
        if values[i] == values[i - 1] and values[i] == values[i + 1]:
            keep[i] = False
    return [key_builder(times[i], values[i]) for i in range(n) if keep[i]]


def build_animation(scene, clip, setup, tr_name, bone_index, fps=FPS):
    sampler, stop, _ = decode_clip(clip)
    n_frames = max(1, int(round(stop * fps)))
    times = [i / fps for i in range(n_frames + 1)]

    keys0 = clip_overrides(clip, scene["hash2tr"], sampler, 0.0)
    animated = [tr for tr in keys0 if tr in setup]

    series = {tr: [] for tr in animated}
    for t in times:
        ov = clip_overrides(clip, scene["hash2tr"], sampler, t)
        for tr in animated:
            base = scene["TR"][tr]
            o = ov.get(tr, {})
            pos = o.get("pos", base["pos"])
            rot = o.get("rot", base["rot"])
            scale = o.get("scale", base["scale"])
            series[tr].append(decompose2d(local_matrix(pos, rot, scale)))

    bones_out = {}
    for tr in animated:
        s0 = setup[tr]
        frames = series[tr]
        track = {}

        rot_vals = []
        prev = 0.0
        for f in frames:
            off = norm180(f["rotation"] - s0["rotation"])
            while off - prev > 180.0:
                off -= 360.0
            while off - prev < -180.0:
                off += 360.0
            rot_vals.append(r2(off))
            prev = off
        if any(abs(v) > EPS for v in rot_vals):
            track["rotate"] = reduce_keys(
                times, rot_vals, lambda t, v: {"time": r2(t), "angle": v},
            )

        tx = [r2(f["x"] - s0["x"]) for f in frames]
        ty = [r2(f["y"] - s0["y"]) for f in frames]
        if any(abs(v) > EPS for v in tx) or any(abs(v) > EPS for v in ty):
            xy = list(zip(tx, ty))
            track["translate"] = reduce_keys(
                times, xy, lambda t, v: {"time": r2(t), "x": v[0], "y": v[1]},
            )

        def factor(cur, base):
            return cur / base if abs(base) > 1e-6 else 1.0

        sx = [r2(factor(f["scaleX"], s0["scaleX"])) for f in frames]
        sy = [r2(factor(f["scaleY"], s0["scaleY"])) for f in frames]
        if any(abs(v - 1) > EPS for v in sx) or any(abs(v - 1) > EPS for v in sy):
            xy = list(zip(sx, sy))
            track["scale"] = reduce_keys(
                times, xy, lambda t, v: {"time": r2(t), "x": v[0], "y": v[1]},
            )

        shy = [r2(norm180(f["shearY"] - s0["shearY"])) for f in frames]
        if any(abs(v) > EPS for v in shy):
            track["shear"] = reduce_keys(
                times, shy, lambda t, v: {"time": r2(t), "x": 0.0, "y": v},
            )

        if track:
            bones_out[tr_name[tr]] = track

    anim_out: dict[str, Any] = {"bones": bones_out}
    slots_out = build_slot_opacity_tracks(scene, clip, sampler, times)
    if slots_out:
        anim_out["slots"] = slots_out
    return anim_out


def write_atlas(scene, out_dir, editor=False):
    atlases = scene["atlases"]
    lines = []
    img_dir = out_dir / "images" if editor else out_dir
    img_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(atlases.keys(), key=lambda k: (len(k), k)):
        page = atlases[key]
        w, h = page["width"], page["height"]
        src_name = page["name"]
        page_name = f"{src_name}.png"
        atlas_page = f"images/{page_name}" if editor else page_name
        page["image"].save(img_dir / page_name)
        lines += [
            "",
            atlas_page,
            f"size: {w},{h}",
            "format: RGBA8888",
            "filter: Linear,Linear",
            "repeat: none",
            src_name,
            "  rotate: false",
            "  xy: 0, 0",
            f"  size: {w}, {h}",
            f"  orig: {w}, {h}",
            "  offset: 0, 0",
            "  index: -1",
        ]
    (out_dir / "skeleton.atlas").write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")


def world_scale(minx, maxx):
    if "SPINE_SCALE" in os.environ:
        return float(os.environ["SPINE_SCALE"])
    if TARGET_W <= 0:
        return 1.0
    w = maxx - minx
    return TARGET_W / w if w > EPS else 1.0


def export_spine(scene, out_dir: Path, editor=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    bones, tr_name, bone_index, setup = build_bones(scene)
    slots, skins = build_slots_and_skin(scene, tr_name, bone_index, editor=editor)

    world = make_world(scene["TR"])
    positions = skin_all(scene["parts"], world)
    minx, miny, maxx, maxy = bounds_of(positions, pad=0.0)
    scale = world_scale(minx, maxx)
    if abs(scale - 1.0) > EPS:
        bones[0]["scaleX"] = r2(scale)
        bones[0]["scaleY"] = r2(scale)

    animations = {}
    clips = [o.read() for o in scene["objs"] if o.type.name == "AnimationClip"]
    for clip in clips:
        try:
            animations[clip.m_Name] = build_animation(
                scene, clip, setup, tr_name, bone_index,
            )
        except Exception as e:
            print(f"  !! animation {clip.m_Name} failed: {e}")

    skel = {
        "skeleton": {
            "hash": "unity2spine",
            "spine": SPINE_VERSION,
            "x": r2(minx * scale),
            "y": r2(miny * scale),
            "width": r2((maxx - minx) * scale),
            "height": r2((maxy - miny) * scale),
            "images": "./images/" if editor else "./",
            "audio": "",
        },
        "bones": bones,
        "slots": slots,
        "skins": skins,
        "animations": animations,
    }

    out_json = out_dir / "skeleton.json"
    out_json.write_text(json.dumps(skel, separators=(",", ":")), encoding="utf-8")
    write_atlas(scene, out_dir, editor=editor)

    mode = "editor" if editor else "runtime"
    print(
        f"[{mode}] bones={len(bones)} slots={len(slots)} animations={len(animations)}"
        f"  scale={r2(scale)} (target_w={TARGET_W})"
    )
    for name, anim in animations.items():
        n_slots = len(anim.get("slots", {}))
        extra = f", {n_slots} animated slots" if n_slots else ""
        print(f"  anim {name}: {len(anim['bones'])} animated bones{extra}")
    print("wrote", out_json)
    print("wrote", out_dir / "skeleton.atlas")


def resolve_output(src: Path, output: Path | None, editor: bool) -> Path:
    if output is not None:
        return output
    parent = src.parent if src.is_file() else src.parent
    return parent / ("spine_editor" if editor else "spine")


def resolve_gif_dir(src: Path, output: Path | None) -> Path:
    if output is not None:
        return output / "gifs"
    parent = src.parent if src.is_file() else src.parent
    return parent / "gifs"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "src",
        type=Path,
        help="Unity AssetBundle path (e.g. …/__data)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="output directory (default: <bundle-parent>/spine_editor, or spine with --runtime)",
    )
    p.add_argument(
        "--editor",
        action="store_true",
        help="Spine Editor import layout (default)",
    )
    p.add_argument(
        "--runtime",
        action="store_true",
        help="runtime export layout (images beside atlas, default dir: spine/)",
    )
    p.add_argument(
        "--both",
        action="store_true",
        help="write runtime and editor exports",
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help="also export AnimationClip previews as GIF",
    )
    p.add_argument(
        "--gif-only",
        action="store_true",
        help="export GIFs only (skip skeleton.json / atlas)",
    )
    p.add_argument(
        "--gif-width",
        type=int,
        default=GIF_W,
        metavar="PX",
        help=f"GIF canvas width (default: {GIF_W})",
    )
    p.add_argument(
        "--gif-fps",
        type=int,
        default=GIF_FPS,
        help=f"GIF frame rate (default: {GIF_FPS})",
    )
    p.add_argument(
        "--gif-bg",
        default=GIF_BG,
        metavar="RRGGBB",
        help=f"GIF background colour (default: {GIF_BG})",
    )
    p.add_argument(
        "--gif-clips",
        default=os.environ.get("GIF_CLIPS"),
        metavar="NAMES",
        help="comma-separated clip name filter (substring match)",
    )
    p.add_argument(
        "--gif-workers",
        type=int,
        default=GIF_WORKERS,
        metavar="N",
        help="parallel part render threads (0=auto, default: auto up to 8)",
    )
    args = p.parse_args(argv)

    src = args.src.resolve()
    if not src.exists():
        p.error(f"source not found: {src}")

    print(f"loading {src}")
    scenes = load_scene(src)
    if len(scenes) > 1:
        print(f"  {len(scenes)} character variants: {', '.join(n for n, _ in scenes)}")

    spine_export = os.environ.get("SPINE_EXPORT", "").lower()
    if args.both:
        modes = (False, True)
    elif args.runtime or spine_export == "runtime":
        modes = (False,)
    else:
        modes = (True,)  # default: editor

    for variant_name, scene in scenes:
        if not scene["parts"]:
            label = variant_name or "(default)"
            print(f"  !! variant {label} skipped: no renderable parts")
            continue
        if args.gif or args.gif_only:
            gif_dir = resolve_gif_dir(src, args.output)
            if variant_name is not None:
                gif_dir = gif_dir / variant_name
            export_gifs(
                scene, gif_dir,
                clip_filter=args.gif_clips,
                target_w=args.gif_width,
                fps=args.gif_fps,
                bg=args.gif_bg,
                workers=args.gif_workers,
            )

        if args.gif_only:
            continue

        for editor in modes:
            out = resolve_output(src, args.output, editor)
            if args.both and editor:
                out = out.parent / "spine_editor" if args.output is None else out
            elif args.both and not editor and args.output is None:
                out = out.parent / "spine"
            if variant_name is not None:
                out = out / variant_name
            export_spine(scene, out, editor=editor)


if __name__ == "__main__":
    main()
