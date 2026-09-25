#!/usr/bin/env python3
"""Unity AssetBundle -> Spine GUI Converter

A PyQt5 GUI tool for converting Unity 2D AssetBundles to Spine 3.8 format,
with batch export and built-in animation preview.

Requirements: pip install PyQt5 PyOpenGL
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import traceback
import zlib
from bisect import bisect_right
from ctypes import c_void_p, sizeof, c_float, c_uint, POINTER, Structure
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import UnityPy
from PIL import Image
from UnityPy.helpers.MeshHelper import MeshHandler

# ---------------------------------------------------------------------------
# PyQt5 + OpenGL imports
# ---------------------------------------------------------------------------
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QUrl, QRectF,
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
    QTransform, QPolygonF, QPainterPath, QOpenGLContext, QSurfaceFormat,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit, QLineEdit,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QSlider, QMessageBox,
    QStyleFactory, QFrame, QScrollArea, QSizePolicy, QOpenGLWidget,
)

# OpenGL
try:
    from OpenGL import GL as gl
    from OpenGL.GL import shaders as gl_shaders
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

# =============  Import core logic from unity_to_spine  =============
# The script is imported dynamically to avoid duplication
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from unity_to_spine import (
    SPINE_VERSION, FPS, TARGET_W, PAD, GIF_W, GIF_FPS, GIF_BG, EPS,
    ATLAS_SUFFIX_RE, GO_TYPE, GO_ACTIVE_ATTR, SMR_TYPE, SMR_COLOR_A_ATTR,
    SMR_BLEND_CROSSFADE, MOUTH_IDLE_PART, MOUTH_A_PART, VIS_EPS,
    mat4, quat_to_m3, make_world, local_matrix, norm180, decompose2d,
    trs_to_spine, r2, atlas_key_from_name, skin_part, skin_all,
    bounds_of, make_canvas_tf, atlases_bgra, prepare_raster_cache,
    rasterize, curve_size, decode_clip, clip_overrides, path_hash,
    sample_clip_properties, part_opacities, load_atlases, load_scene,
    build_bones, part_weighted_vertices, mesh_image_size,
    build_spine_mesh_attachment, part_slot_names,
    build_slots_and_skin, build_animation, write_atlas, world_scale,
    export_spine, export_gif, export_gifs,
)


# =============  Preview Renderer (based on viewer_server logic)  =============

PREVIEW_RENDER_W = 1024
PREVIEW_MIN_RENDER_W = 512
PREVIEW_MAX_RENDER_W = 2048
PREVIEW_FPS = 15


# ---------------------------------------------------------------------------
# OpenGL shader sources for GPU-accelerated Spine mesh rendering
# ---------------------------------------------------------------------------
_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 aPos;         // vertex position in bone-local space
layout(location = 1) in vec2 aTexCoord;    // atlas UV
layout(location = 2) in vec4 aBoneIndices; // 4 bone indices (int-as-float)
layout(location = 3) in vec4 aBoneWeights; // 4 bone weights

uniform mat4 uMVP;                          // model-view-projection
uniform mat4 uBoneMats[256];                // per-bone world matrices
uniform float uOpacity;                     // per-part opacity

out vec2 vTexCoord;
out float vOpacity;

void main() {
    // GPU skinning: weighted blend of up to 4 bones
    mat4 skinMat = uBoneMats[int(aBoneIndices.x)] * aBoneWeights.x
                 + uBoneMats[int(aBoneIndices.y)] * aBoneWeights.y
                 + uBoneMats[int(aBoneIndices.z)] * aBoneWeights.z
                 + uBoneMats[int(aBoneIndices.w)] * aBoneWeights.w;

    vec4 worldPos = skinMat * vec4(aPos, 0.0, 1.0);
    gl_Position = uMVP * worldPos;
    vTexCoord = aTexCoord;
    vOpacity = uOpacity;
}
"""

_FRAGMENT_SHADER = """
#version 330 core

in vec2 vTexCoord;
in float vOpacity;

uniform sampler2D uTexture;

out vec4 fragColor;

void main() {
    vec4 texColor = texture(uTexture, vTexCoord);
    fragColor = vec4(texColor.rgb, texColor.a * vOpacity);
}
"""


class GPURenderer:
    """GPU-accelerated Spine mesh renderer using OpenGL.

    Each part's atlas region + triangles are uploaded once as VBOs + texture.
    At render time, only bone matrices and opacity are updated per frame –
    the GPU does all skinning and rasterization in hardware.

    All GL uniform locations are cached at init to avoid the extremely
    expensive per-frame/per-part glGetUniformLocation() string lookups.
    """

    def __init__(self):
        self._ctx: Optional[QOpenGLContext] = None
        self._surface: Optional[Any] = None  # QOffscreenSurface
        self._fbo: Optional[int] = None       # OpenGL framebuffer object ID
        self._fbo_tex: Optional[int] = None   # color attachment texture
        self._fbo_depth: Optional[int] = None # depth renderbuffer
        self._fbo_w = 0
        self._fbo_h = 0
        self._shader_prog: Optional[int] = None
        self._vao: Optional[int] = None
        self._initialized = False

        # ---- Cached uniform locations (set once during init_gl) ----
        self._loc_uMVP: int = -1
        self._loc_uBoneMats: int = -1
        self._loc_uOpacity: int = -1
        self._loc_uTexture: int = -1

        # Per-part GPU data
        # _part_gpu[i] = {
        #     "vbo": GL buffer ID (packed: pos2+uv2+indices4+weights4 = 12 floats/vert),
        #     "ebo": GL element buffer ID,
        #     "texture": GL texture ID,
        #     "vertex_count": int,  # total vertices
        #     "index_count": int,   # total indices (triangles * 3)
        # }
        self._part_gpu: list[dict] = []
        # Per-bone index mapping: bone_tr_id -> index in uBoneMats[]
        self._bone_to_gpu_index: dict = {}  # tr_id -> 0-based index
        self._gpu_bone_count = 0

    def init_gl(self):
        """Must be called on the GL thread / with a valid context."""
        if not HAS_OPENGL:
            return False
        try:
            # Compile shaders
            vs = gl_shaders.compileShader(_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
            fs = gl_shaders.compileShader(_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
            # validate=False: glValidateProgram runs before any VAO/texture is
            # bound, which some drivers spuriously fail even though the
            # program links and runs fine. Linking is still checked below.
            self._shader_prog = gl_shaders.compileProgram(vs, fs, validate=False)
            gl.glDeleteShader(vs)
            gl.glDeleteShader(fs)

            # Create a global VAO
            self._vao = gl.glGenVertexArrays(1)

            # ---- Cache ALL uniform locations ONCE (never call glGetUniformLocation per frame!) ----
            gl.glUseProgram(self._shader_prog)
            self._loc_uMVP = gl.glGetUniformLocation(self._shader_prog, "uMVP")
            self._loc_uBoneMats = gl.glGetUniformLocation(self._shader_prog, "uBoneMats")
            self._loc_uOpacity = gl.glGetUniformLocation(self._shader_prog, "uOpacity")
            self._loc_uTexture = gl.glGetUniformLocation(self._shader_prog, "uTexture")
            gl.glUseProgram(0)

            self._initialized = True
            return True
        except Exception as e:
            print(f"[GL] Shader init failed: {e}")
            return False

    def upload_geometry(self, engine: "PreviewEngine"):
        """Upload all parts as GPU buffers and textures. Call once after loading."""
        if not self._initialized:
            return

        # Clean up previous
        for pg in self._part_gpu:
            gl.glDeleteBuffers(1, [pg["vbo"]])
            gl.glDeleteBuffers(1, [pg["ebo"]])
            gl.glDeleteTextures([pg["texture"]])
        self._part_gpu.clear()
        self._bone_to_gpu_index.clear()
        self._gpu_bone_count = 0

        # Build bone → GPU index mapping (only bones that appear in parts)
        bone_set: set = set()
        for entry in engine.raster_cache:
            part = entry["part"]
            if part["kind"] == "mesh" and not part["single"]:
                for b in part["bones"]:
                    bone_set.add(b)
            elif part["kind"] == "mesh":
                bone_set.add(part["bones"][0])
            elif part["kind"] == "sprite":
                bone_set.add(part["tr_pid"])

        bone_list = sorted(bone_set)
        self._bone_to_gpu_index = {b: i for i, b in enumerate(bone_list)}
        self._gpu_bone_count = len(bone_list)

        for entry in engine.raster_cache:
            part = entry["part"]
            crop = entry["atlas_crop"]  # BGRA uint8 HxWx4
            ch, cw = crop.shape[:2]

            # Upload texture (BGRA -> RGBA)
            tex_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            # Convert BGRA to RGBA for OpenGL
            rgba_data = np.ascontiguousarray(crop[:, :, [2, 1, 0, 3]])
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                cw, ch, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba_data,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            # Build vertex data
            src_local = entry["src_local"]  # atlas-local pixel coords
            faces = entry["faces"]

            n_verts = len(src_local)
            # atlas UV: normalize to [0,1]
            uvs = src_local.copy()
            uvs[:, 0] /= cw
            uvs[:, 1] /= ch
            # flip V for OpenGL
            uvs[:, 1] = 1.0 - uvs[:, 1]

            # Bone indices and weights
            if part["kind"] == "mesh":
                bi = part.get("bi")  # (n_verts, k) int
                bw = part.get("bw")  # (n_verts, k) float
                if bi is not None and bw is not None:
                    k = bi.shape[1]
                    gpu_bi = np.zeros((n_verts, 4), dtype=np.float32)
                    gpu_bw = np.zeros((n_verts, 4), dtype=np.float32)
                    for j in range(min(k, 4)):
                        for vi in range(n_verts):
                            b_tr_id = part["bones"][int(bi[vi, j])]
                            gpu_idx = self._bone_to_gpu_index.get(b_tr_id, 0)
                            gpu_bi[vi, j] = float(gpu_idx)
                            gpu_bw[vi, j] = float(bw[vi, j])
                    # Normalize weights
                    wsum = gpu_bw.sum(axis=1, keepdims=True) + 1e-10
                    gpu_bw /= wsum
                else:
                    # Single bone
                    b_tr_id = part["bones"][0]
                    gpu_idx = self._bone_to_gpu_index.get(b_tr_id, 0)
                    gpu_bi = np.full((n_verts, 4), float(gpu_idx), dtype=np.float32)
                    gpu_bw = np.zeros((n_verts, 4), dtype=np.float32)
                    gpu_bw[:, 0] = 1.0
            else:
                # Sprite: single bone
                b_tr_id = part["tr_pid"]
                gpu_idx = self._bone_to_gpu_index.get(b_tr_id, 0)
                gpu_bi = np.full((n_verts, 4), float(gpu_idx), dtype=np.float32)
                gpu_bw = np.zeros((n_verts, 4), dtype=np.float32)
                gpu_bw[:, 0] = 1.0

            # Build interleaved vertex buffer:
            # [pos.x, pos.y, uv.x, uv.y, bone0, bone1, bone2, bone3, weight0, weight1, weight2, weight3] = 12 floats
            vbo_data = np.zeros((n_verts, 12), dtype=np.float32)
            # positions are zero in bone-local space for Spine-style (world transform applied via bone mat)
            # Actually for Unity we have world-space vertices in src_local, but we need bone-local.
            # The skinning shader applies bone world matrices to local positions.
            # For simplicity, we use (0,0) as bone-local pos and rely on bone matrices.
            # BUT the mesh vertices ARE in bone-local already from the bind pose.
            # Let's use the vh data: vh = [x, y, z, 1] in bind-pose local space.
            if part["kind"] == "mesh":
                vh = part["vh"]  # (n_verts, 4) homogeneous local coords
                vbo_data[:, 0] = vh[:, 0].astype(np.float32)
                vbo_data[:, 1] = vh[:, 1].astype(np.float32)
            else:
                # sprite: sv2[:, :2] are already in local space
                sv2 = part["sv2"]
                vbo_data[:, 0] = sv2[:, 0].astype(np.float32)
                vbo_data[:, 1] = sv2[:, 1].astype(np.float32)

            vbo_data[:, 2] = uvs[:, 0]
            vbo_data[:, 3] = uvs[:, 1]
            vbo_data[:, 4:8] = gpu_bi
            vbo_data[:, 8:12] = gpu_bw

            # Index buffer: flatten triangle faces
            indices = faces.astype(np.uint32).ravel()
            # Ensure indices are within range
            indices = np.clip(indices, 0, n_verts - 1)

            # Create VBO
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vbo_data.nbytes, vbo_data, gl.GL_STATIC_DRAW)

            # Create EBO
            ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

            self._part_gpu.append({
                "vbo": vbo,
                "ebo": ebo,
                "texture": tex_id,
                "index_count": len(indices),
                "vertex_count": n_verts,
            })

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _ensure_fbo(self, w: int, h: int):
        """Create or resize the offscreen FBO."""
        if self._fbo is not None and self._fbo_w == w and self._fbo_h == h:
            return
        if self._fbo is not None:
            gl.glDeleteFramebuffers(1, [self._fbo])
            gl.glDeleteTextures([self._fbo_tex])
            if self._fbo_depth:
                gl.glDeleteRenderbuffers(1, [self._fbo_depth])

        self._fbo_w = w
        self._fbo_h = h

        # Color texture
        self._fbo_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._fbo_tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # Depth renderbuffer
        self._fbo_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._fbo_depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, w, h)

        # FBO
        self._fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._fbo_tex, 0,
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, self._fbo_depth,
        )

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"[GL] FBO incomplete: {status}")

    def render_gpu(
        self,
        engine: "PreviewEngine",
        bone_matrices: np.ndarray,  # (N, 4, 4) world matrices, indexed by _bone_to_gpu_index
        opacities: list[float],
        canvas_w: int,
        canvas_h: int,
        bg_color=(0.05, 0.05, 0.10),
    ) -> Optional[np.ndarray]:
        """Render all parts to a BGRA numpy array using GPU.

        bone_matrices: world-space 4x4 matrices for each bone in _bone_to_gpu_index order.
        opacities: per-part opacity [0, 1].
        canvas_w, canvas_h: output resolution.
        Returns BGRA uint8 numpy array, or None on failure.
        """
        if not self._initialized or not self._part_gpu:
            return None

        self._ensure_fbo(canvas_w, canvas_h)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glViewport(0, 0, canvas_w, canvas_h)

        # Clear
        gl.glClearColor(*bg_color, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(
            gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,
            gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA,
        )

        gl.glUseProgram(self._shader_prog)
        gl.glBindVertexArray(self._vao)

        # Build ortho projection: map world coords to [-1,1]
        # World bounds: (minx, miny) -> (maxx, maxy) maps to canvas
        minx, miny = engine.minx, engine.miny
        maxx, maxy = engine.maxx, engine.maxy
        ww = maxx - minx
        wh = maxy - miny
        if ww < 1e-6 or wh < 1e-6:
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            return None

        # Orthographic projection: world -> NDC
        proj = np.array([
            [2.0 / ww, 0, 0, -(maxx + minx) / ww],
            [0, 2.0 / wh, 0, -(maxy + miny) / wh],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # OpenGL uses column-major; use cached uniform location
        gl.glUniformMatrix4fv(self._loc_uMVP, 1, gl.GL_TRUE, proj.T.copy())

        # Upload bone matrices (flattened, column-major) – cached location
        bone_flat = np.ascontiguousarray(bone_matrices.transpose(0, 2, 1).reshape(-1), dtype=np.float32)
        gl.glUniformMatrix4fv(self._loc_uBoneMats, len(bone_matrices), gl.GL_TRUE, bone_flat)

        # Pre-compute stride & attribute offsets (same for all parts)
        stride = 12 * 4  # 12 floats * 4 bytes
        c_void_0 = c_void_p(0)
        c_void_8 = c_void_p(8)
        c_void_16 = c_void_p(16)
        c_void_32 = c_void_p(32)

        for i, pg in enumerate(self._part_gpu):
            if i >= len(opacities):
                opacity = 1.0
            else:
                opacity = max(0.0, min(1.0, opacities[i]))
            if opacity < 0.001:
                continue

            # Set opacity uniform – cached location
            gl.glUniform1f(self._loc_uOpacity, opacity)

            # Bind texture – cached location
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, pg["texture"])
            gl.glUniform1i(self._loc_uTexture, 0)

            # Bind VBO and set vertex attributes
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pg["vbo"])

            # aPos (location 0): 2 floats at offset 0
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_0)
            gl.glEnableVertexAttribArray(0)
            # aTexCoord (location 1): 2 floats at offset 8
            gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_8)
            gl.glEnableVertexAttribArray(1)
            # aBoneIndices (location 2): 4 floats at offset 16
            gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_16)
            gl.glEnableVertexAttribArray(2)
            # aBoneWeights (location 3): 4 floats at offset 32
            gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_32)
            gl.glEnableVertexAttribArray(3)

            # Bind EBO and draw
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, pg["ebo"])
            gl.glDrawElements(gl.GL_TRIANGLES, pg["index_count"], gl.GL_UNSIGNED_INT, c_void_0)

        # Cleanup
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
        gl.glDisableVertexAttribArray(3)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_BLEND)

        # Read pixels from FBO
        pixels = gl.glReadPixels(0, 0, canvas_w, canvas_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(canvas_h, canvas_w, 4)
        # OpenGL reads bottom-to-top, flip vertically
        img = np.ascontiguousarray(img[::-1])
        # RGBA -> BGRA (for consistency with cv2 format)
        img_bgra = np.ascontiguousarray(img[:, :, [2, 1, 0, 3]])

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return img_bgra

    def cleanup(self):
        """Release all GPU resources."""
        if not self._initialized:
            return
        for pg in self._part_gpu:
            gl.glDeleteBuffers(2, [pg["vbo"], pg["ebo"]])
            gl.glDeleteTextures([pg["texture"]])
        self._part_gpu.clear()
        if self._fbo is not None:
            gl.glDeleteFramebuffers(1, [self._fbo])
            gl.glDeleteTextures([self._fbo_tex])
            if self._fbo_depth:
                gl.glDeleteRenderbuffers(1, [self._fbo_depth])
            self._fbo = None
        if self._vao is not None:
            gl.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._shader_prog is not None:
            gl.glDeleteProgram(self._shader_prog)
            self._shader_prog = None
        self._initialized = False


class PreviewEngine:
    """Lightweight preview engine: loads scene + exports, rasterizes frames.

    Performance strategy:
    - Pre-samples all animation clip data at PREVIEW_FPS when loading.
    - Stores bone overrides, go_active, smr_alpha, smr_props per frame
      as simple dict/list structures (no Unity objects involved).
    - At render time, linearly interpolates between two cached frames,
      then only does the remaining work: make_world → build GPU bone mats → GPU render.
    - GPU renderer (GPURenderer) does hardware-accelerated skinning + rasterization.
    """

    def __init__(self):
        self.scene: Optional[dict] = None
        self.variants: list = []  # [(variant_name_or_None, scene_dict), ...] from load_scene()
        self.variant_index: int = 0
        self.skel: Optional[dict] = None
        self.bones: list = []
        self.tr_name: dict = {}
        self.bone_index: dict = {}
        self.setup: dict = {}
        self.parts: list = []
        self.atlases: dict = {}
        self.atlas_bgra: dict = {}
        self.raster_cache: list = []
        self.animations: dict = {}
        self.current_anim: Optional[str] = None
        self.anim_sampler = None
        self.anim_stop = 0.0
        self.time = 0.0
        self.to_canvas = None
        self.W = 0
        self.H = 0
        self.scale_val = 1.0
        self.minx = 0.0
        self.miny = 0.0
        self.maxx = 0.0
        self.maxy = 0.0
        self.show_skeleton = True
        self.slot_visibility: dict[str, bool] = {}
        self.bg_color = (13, 13, 26)  # dark background
        # ---- performance caches ----
        self._clip_cache: dict[str, Any] = {}      # clip_name -> deserialized clip
        self._bone_tr_map: Optional[dict] = None     # bone_name -> tr_id
        self._bone_parent_tr_map: Optional[dict] = None  # bone_name -> parent_tr_id
        # ---- pre-sampled frame cache ----
        self._frame_cache: dict[str, dict] = {}
        self._current_frame_cache: Optional[dict] = None
        # ---- GPU renderer ----
        self.gpu: Optional[GPURenderer] = None
        self._gpu_bone_mats: Optional[np.ndarray] = None
        self._gpu_opacities: Optional[list[float]] = None
        self._gpu_skeleton_lines: list = []
        self._gpu_frame_version: int = 0  # incremented each time data is updated

    def load_bundle(self, src: Path, variant_index: int = 0) -> str:
        """Load a Unity bundle and prepare for preview.

        A bundle may pack several skins (N, NS1, NS2, ...) into one file;
        load_scene() returns one (name, scene) pair per skin. The preview
        can only show one at a time, so this shows `variant_index` (the
        first by default) and reports how many others were found.
        """
        scenes = load_scene(src)
        self.variants = scenes
        self.variant_index = variant_index
        variant_name, scene = scenes[variant_index]
        self.scene = scene
        self.atlases = scene["atlases"]
        self.parts = scene["parts"]

        # Build skeleton
        bones, tr_name, bone_index, setup = build_bones(scene)
        self.bones = bones
        self.tr_name = tr_name
        self.bone_index = bone_index
        self.setup = setup

        # Build slots
        slots, skins = build_slots_and_skin(scene, tr_name, bone_index, editor=False)

        # Compute bounds
        world = make_world(scene["TR"])
        positions = skin_all(scene["parts"], world)
        minx, miny, maxx, maxy = bounds_of(positions, pad=0.0)
        scale = world_scale(minx, maxx)
        self.scale_val = scale
        self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy

        # Prepare raster cache
        self.atlas_bgra = atlases_bgra(scene["atlases"])
        self.raster_cache = prepare_raster_cache(scene["parts"], self.atlas_bgra)

        # ---- Cache all AnimationClip objects ONCE ----
        self._clip_cache = {}
        for o in scene["objs"]:
            if o.type.name == "AnimationClip":
                clip = o.read()
                self._clip_cache[clip.m_Name] = clip

        # Build Spine animations
        animations = {}
        for name, clip in self._clip_cache.items():
            try:
                animations[name] = build_animation(
                    scene, clip, setup, tr_name, bone_index,
                )
            except Exception:
                pass
        self.animations = animations

        # ---- Pre-sample all animation frames for fast playback ----
        self._frame_cache = {}
        for name, clip in self._clip_cache.items():
            try:
                self._frame_cache[name] = self._pre_sample_animation(scene, clip)
            except Exception:
                pass

        # ---- Pre-compute bone TR lookups (avoid per-frame dict iteration) ----
        self._bone_tr_map = {}
        self._bone_parent_tr_map = {}
        for tr_id, nm in tr_name.items():
            self._bone_tr_map[nm] = tr_id
        for b in bones[1:]:
            bname = b["name"]
            parent_name = b.get("parent", "root")
            self._bone_tr_map.setdefault(bname, None)
            if parent_name and parent_name != "root":
                self._bone_parent_tr_map[bname] = self._bone_tr_map.get(parent_name)

        # Default slot visibility
        self.slot_visibility = {}
        for slot in slots:
            self.slot_visibility[slot["name"]] = True

        self.skel = {
            "skeleton": {
                "spine": SPINE_VERSION,
                "x": r2(minx * scale),
                "y": r2(miny * scale),
                "width": r2((maxx - minx) * scale),
                "height": r2((maxy - miny) * scale),
            },
            "bones": bones,
            "slots": slots,
            "skins": skins,
            "animations": animations,
        }

        # Setup canvas transform
        target = PREVIEW_RENDER_W
        bounds = (minx, miny, maxx, maxy)
        self.to_canvas, self.W, self.H = make_canvas_tf(bounds, target)

        # ---- Initialize GPU renderer ----
        # GPU upload happens later on the GL thread via _upload_to_gpu()
        self.gpu = GPURenderer()
        self._gpu_upload_pending = True

        # Select first animation
        anim_names = list(animations.keys())
        if anim_names:
            self.select_animation(anim_names[0])

        msg = (f"骨骼: {len(bones)}  插槽: {len(slots)}  "
               f"动画: {len(animations)}  版本: {SPINE_VERSION}")
        if len(scenes) > 1:
            names = "、".join((n or "?") for n, _ in scenes)
            msg += f"  [此文件含 {len(scenes)} 套皮肤: {names}；当前预览: {variant_name or '(默认)'}]"
        return msg

    def _pre_sample_animation(self, scene: dict, clip) -> dict:
        """Pre-sample all frames of an animation at PREVIEW_FPS.

        Returns a dict with cached per-frame data to avoid Unity curve
        evaluation at render time.
        """
        sampler, stop_time, _ = decode_clip(clip)
        hash2tr = scene["hash2tr"]
        n_frames = max(1, int(stop_time * PREVIEW_FPS))
        times = np.array([i / PREVIEW_FPS for i in range(n_frames + 1)], dtype=np.float64)
        # Clamp last time to stop_time
        if times[-1] > stop_time:
            times[-1] = stop_time

        overrides_list = []
        go_active_list = []
        smr_alpha_list = []
        smr_props_list = []
        opacities_list = []

        for t in times:
            ov = clip_overrides(clip, hash2tr, sampler, float(t))
            go_active, smr_alpha, smr_props = sample_clip_properties(clip, sampler, float(t))
            opacities = part_opacities(scene, go_active, smr_alpha, smr_props)
            overrides_list.append(ov)
            go_active_list.append(go_active)
            smr_alpha_list.append(smr_alpha)
            smr_props_list.append(smr_props)
            opacities_list.append(opacities)

        return {
            "times": times,
            "overrides": overrides_list,
            "go_active": go_active_list,
            "smr_alpha": smr_alpha_list,
            "smr_props": smr_props_list,
            "opacities": opacities_list,
        }

    def _interpolate_cached_frame(self, t: float) -> tuple:
        """Given time t, interpolate between two pre-sampled frames.

        Returns (overrides, go_active, smr_alpha, smr_props, opacities)
        using linear interpolation between cached samples.
        """
        cache = self._current_frame_cache
        if cache is None:
            return {}, {}, {}, {}, []
        times = cache["times"]
        n = len(times)
        if n == 0:
            return {}, {}, {}, {}, []

        # Binary search for the right interval
        t = max(0.0, min(t, times[-1]))
        idx = bisect_right(times, t) - 1
        idx = max(0, min(idx, n - 1))

        if idx >= n - 1 or abs(t - times[idx]) < 1e-9:
            # Exact or last frame
            return (
                cache["overrides"][idx],
                cache["go_active"][idx],
                cache["smr_alpha"][idx],
                cache["smr_props"][idx],
                cache["opacities"][idx],
            )

        # Interpolate between idx and idx+1
        t0, t1 = times[idx], times[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        alpha = max(0.0, min(1.0, alpha))
        beta = 1.0 - alpha

        # ---- Interpolate overrides (per-bone pos/rot/scale) ----
        ov0 = cache["overrides"][idx]
        ov1 = cache["overrides"][idx + 1]
        ov = {}
        all_keys = set(ov0.keys()) | set(ov1.keys())
        for tr_id in all_keys:
            a = ov0.get(tr_id, {})
            b = ov1.get(tr_id, {})
            merged = {}
            for attr in ("pos", "rot", "scale"):
                va = a.get(attr)
                vb = b.get(attr)
                if va is not None and vb is not None:
                    merged[attr] = np.array(va) * beta + np.array(vb) * alpha
                elif va is not None:
                    merged[attr] = va
                elif vb is not None:
                    merged[attr] = vb
            ov[tr_id] = merged

        # ---- Interpolate opacities (per-part float) ----
        op0 = cache["opacities"][idx]
        op1 = cache["opacities"][idx + 1]
        opacities = [o0 * beta + o1 * alpha for o0, o1 in zip(op0, op1)]

        # For go_active / smr_alpha / smr_props, use the closer frame
        # (these are mostly boolean-ish or slowly changing)
        if alpha < 0.5:
            go_active = cache["go_active"][idx]
            smr_alpha = cache["smr_alpha"][idx]
            smr_props = cache["smr_props"][idx]
        else:
            go_active = cache["go_active"][idx + 1]
            smr_alpha = cache["smr_alpha"][idx + 1]
            smr_props = cache["smr_props"][idx + 1]

        return ov, go_active, smr_alpha, smr_props, opacities

    def select_animation(self, name: str):
        if name not in self.animations:
            return
        self.current_anim = name
        self._current_frame_cache = self._frame_cache.get(name)
        # Update anim_stop from pre-sampled cache
        cache = self._current_frame_cache
        if cache and len(cache["times"]) > 0:
            self.anim_stop = float(cache["times"][-1])
        else:
            self.anim_stop = 1.0
        self.time = 0.0
        # Clear GPU cached data
        self._gpu_bone_mats = None
        self._gpu_opacities = None
        self._gpu_skeleton_lines = []
        self._gpu_frame_version = 0

    def _prepare_gpu_render_data(self, t: float):
        """Compute bone matrices, opacities, and skeleton lines for GPU rendering.

        Stores results in self._gpu_bone_mats, self._gpu_opacities, and
        self._gpu_skeleton_lines.  All derived from a single
        _interpolate_cached_frame + make_world call, avoiding redundant
        computation that previously happened when render_skeleton_overlay
        was called separately.
        """
        if self.scene is None or self.current_anim is None or self.gpu is None:
            self._gpu_bone_mats = None
            self._gpu_opacities = None
            self._gpu_skeleton_lines = []
            return

        ov, go_active, smr_alpha, smr_props, opacities = self._interpolate_cached_frame(t)

        # Apply slot visibility
        part_names = part_slot_names(self.scene["parts"])
        for i, name in enumerate(part_names):
            if i < len(opacities) and not self.slot_visibility.get(name, True):
                opacities[i] = 0.0

        # Compute world matrices for all bones (single make_world call)
        world_fn = make_world(self.scene["TR"], ov)

        # Build bone world matrices for GPU
        gpu = self.gpu
        n_bones = gpu._gpu_bone_count
        bone_mats = np.zeros((n_bones, 4, 4), dtype=np.float32)
        for tr_id, idx in gpu._bone_to_gpu_index.items():
            if tr_id in self.scene["TR"]:
                bone_mats[idx] = world_fn(tr_id).astype(np.float32)

        self._gpu_bone_mats = bone_mats
        self._gpu_opacities = opacities
        self._gpu_frame_version += 1

        # Compute skeleton overlay from the SAME world_fn (no second make_world!)
        bone_lines = []
        if self.show_skeleton:
            tr_map = self._bone_tr_map or {}
            parent_map = self._bone_parent_tr_map or {}
            for b in self.bones[1:]:  # skip root
                bone_name = b["name"]
                tr_id = tr_map.get(bone_name)
                parent_tr_id = parent_map.get(bone_name)
                if tr_id is None or parent_tr_id is None:
                    continue
                w_child = world_fn(tr_id)
                w_parent = world_fn(parent_tr_id)
                cx, cy = w_child[0, 3], w_child[1, 3]
                px, py = w_parent[0, 3], w_parent[1, 3]
                bone_lines.append((px, py, cx, cy))
        self._gpu_skeleton_lines = bone_lines

    def render_frame(self, t: float, target_w: Optional[int] = None) -> np.ndarray:
        """Render a single frame at time t to a BGRA numpy array.

        Uses GPU-accelerated rendering (OpenGL skinning + rasterization) when
        a GL context is active. Falls back to CPU cv2 rasterize otherwise.
        """
        if self.scene is None:
            return np.zeros((self.H, self.W, 4), dtype=np.uint8)
        # A bundle with no AnimationClip at all (a static prop/accessory)
        # has no current_anim - still render its rest/bind pose rather
        # than an empty frame.

        # Choose render resolution
        if target_w is not None:
            tw = max(PREVIEW_MIN_RENDER_W, min(PREVIEW_MAX_RENDER_W, target_w))
            if tw != self.W:
                to_canvas, W, H = make_canvas_tf(
                    (self.minx, self.miny, self.maxx, self.maxy), tw,
                )
            else:
                to_canvas, W, H = self.to_canvas, self.W, self.H
        else:
            to_canvas, W, H = self.to_canvas, self.W, self.H

        # ---- GPU rendering path (FBO offscreen) ----
        # Use pre-computed data when available, otherwise compute it now
        bone_mats = getattr(self, '_gpu_bone_mats', None)
        opacities = getattr(self, '_gpu_opacities', None)

        if bone_mats is None or opacities is None:
            # Compute fresh data (one _interpolate_cached_frame call)
            ov, go_active, smr_alpha, smr_props, opacities = self._interpolate_cached_frame(t)
            part_names = part_slot_names(self.scene["parts"])
            for i, name in enumerate(part_names):
                if i < len(opacities) and not self.slot_visibility.get(name, True):
                    opacities[i] = 0.0
            world_fn = make_world(self.scene["TR"], ov)

            # Build bone matrices for GPU
            if self.gpu is not None and self.gpu._bone_to_gpu_index:
                gpu = self.gpu
                n_bones = gpu._gpu_bone_count
                bone_mats = np.zeros((n_bones, 4, 4), dtype=np.float32)
                for tr_id, idx in gpu._bone_to_gpu_index.items():
                    if tr_id in self.scene["TR"]:
                        bone_mats[idx] = world_fn(tr_id).astype(np.float32)

        if self.gpu is not None and self.gpu._initialized and self.gpu._part_gpu and bone_mats is not None:
            try:
                frame = self.gpu.render_gpu(
                    self,
                    bone_mats,
                    opacities,
                    W, H,
                    bg_color=(
                        self.bg_color[0] / 255.0,
                        self.bg_color[1] / 255.0,
                        self.bg_color[2] / 255.0,
                    ),
                )
                if frame is not None:
                    return frame
            except Exception as e:
                print(f"[GPU] FBO render failed, falling back to CPU: {e}")

        # ---- CPU fallback (only reached when GPU fails or data missing) ----
        ov, go_active, smr_alpha, smr_props, opacities2 = self._interpolate_cached_frame(t)
        part_names = part_slot_names(self.scene["parts"])
        for i, name in enumerate(part_names):
            if i < len(opacities2) and not self.slot_visibility.get(name, True):
                opacities2[i] = 0.0
        world_fn = make_world(self.scene["TR"], ov)
        positions = skin_all(self.scene["parts"], world_fn)
        frame = rasterize(
            self.raster_cache, positions, to_canvas, W, H,
            # No current animation (a static prop/accessory) means
            # _interpolate_cached_frame() has nothing cached and returns
            # opacities2 = [] - rasterize() indexes opacities[i] per part,
            # so an empty-but-not-None list would raise IndexError; treat
            # it the same as "no override" (full opacity) instead.
            workers=1, opacities=(opacities2 or None),
        )
        return frame

    def render_skeleton_overlay(self, t: float) -> list:
        """Return bone world-space positions for skeleton overlay drawing."""
        if self.scene is None or self.current_anim is None:
            return []
        if self._current_frame_cache is None:
            return []

        # Use interpolated overrides from pre-sampled cache
        ov, _, _, _, _ = self._interpolate_cached_frame(t)
        world = make_world(self.scene["TR"], ov)

        bone_lines = []
        tr_map = self._bone_tr_map or {}
        parent_map = self._bone_parent_tr_map or {}

        for b in self.bones[1:]:  # skip root
            bone_name = b["name"]
            tr_id = tr_map.get(bone_name)
            parent_tr_id = parent_map.get(bone_name)
            if tr_id is None or parent_tr_id is None:
                continue

            w_child = world(tr_id)
            w_parent = world(parent_tr_id)
            cx, cy = w_child[0, 3], w_child[1, 3]
            px, py = w_parent[0, 3], w_parent[1, 3]
            # Return world-space coordinates; paintEvent handles projection
            bone_lines.append((px, py, cx, cy))
        return bone_lines


# =============  Worker Threads  =============

class PreviewOnlyWorker(QThread):
    """Background thread: load Unity Bundle and prepare preview WITHOUT exporting Spine."""
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)  # PreviewEngine

    def __init__(self, src: Path):
        super().__init__()
        self.src = src

    def run(self):
        try:
            self.progress.emit(f"正在加载: {self.src}")
            engine = PreviewEngine()
            msg = engine.load_bundle(self.src)
            self.progress.emit(f"预览就绪: {msg}")
            self.preview_ready.emit(engine)
            self.finished_signal.emit(True, "预览加载成功")
        except Exception as e:
            self.progress.emit(f"错误: {e}")
            traceback.print_exc()
            self.finished_signal.emit(False, str(e))


class ConvertWorker(QThread):
    """Background thread for single file conversion (load + export + preview)."""
    progress = pyqtSignal(str)       # log message
    finished_signal = pyqtSignal(bool, str)  # success, message
    preview_ready = pyqtSignal(object)  # PreviewEngine

    def __init__(self, src: Path, output_dir: Optional[Path], editor: bool):
        super().__init__()
        self.src = src
        self.output_dir = output_dir
        self.editor = editor

    def run(self):
        try:
            self.progress.emit(f"正在加载: {self.src}")
            scenes = load_scene(self.src)

            # Determine output
            if self.output_dir:
                out_base = self.output_dir
            else:
                parent = self.src.parent if self.src.is_file() else self.src.parent
                out_base = parent / ("spine_editor" if self.editor else "spine")

            if len(scenes) > 1:
                self.progress.emit(f"检测到 {len(scenes)} 套皮肤，分别导出到子文件夹")
            for variant_name, scene in scenes:
                out = out_base / variant_name if variant_name else out_base
                self.progress.emit(f"正在导出到: {out}")
                export_spine(scene, out, editor=self.editor)

            # Create preview engine
            engine = PreviewEngine()
            msg = engine.load_bundle(self.src)
            self.progress.emit(f"预览就绪: {msg}")
            self.preview_ready.emit(engine)

            self.finished_signal.emit(True, f"导出成功: {out_base}")
        except Exception as e:
            self.progress.emit(f"错误: {e}")
            traceback.print_exc()
            self.finished_signal.emit(False, str(e))


class BatchConvertWorker(QThread):
    """Background thread for batch conversion."""
    progress = pyqtSignal(str)
    file_progress = pyqtSignal(int, int)  # current, total
    finished_signal = pyqtSignal(int, int)  # success_count, fail_count

    def __init__(self, sources: list[Path], output_base: Path, editor: bool):
        super().__init__()
        self.sources = sources
        self.output_base = output_base
        self.editor = editor

    def _atlas_base_name(self, scene: dict) -> str:
        """Extract the base name from atlas texture names (strip _partN suffix).

        e.g. '3P_BlackWyrm_NS1_part1' -> '3P_BlackWyrm_NS1'
        """
        atlases = scene.get("atlases", {})
        if atlases:
            for key, page in sorted(atlases.items(), key=lambda kv: (len(kv[0]), kv[0])):
                name = page.get("name", "")
                if name:
                    # Remove the trailing _partN to get the base name
                    import re
                    base = re.sub(r"_part\d+$", "", name, flags=re.I)
                    if base:
                        return base
        # Fallback: use file stem
        return "unknown"

    def run(self):
        success = 0
        fail = 0
        total = len(self.sources)
        # Track used folder names to avoid overwriting duplicates
        used_names: dict[str, int] = {}

        for i, src in enumerate(self.sources):
            self.file_progress.emit(i + 1, total)
            try:
                self.progress.emit(f"[{i + 1}/{total}] 正在处理: {src}")
                scenes = load_scene(src)
                # A bundle may pack several skins into one file; each gets
                # its own folder. Prefer the skin's own name when the file
                # was split, otherwise fall back to the atlas texture base
                # name (e.g. "3P_BlackWyrm_NS1") as before. Duplicate names
                # get _2, _3, etc.
                for variant_name, scene in scenes:
                    base_name = variant_name or self._atlas_base_name(scene)
                    if base_name in used_names:
                        used_names[base_name] += 1
                        folder_name = f"{base_name}_{used_names[base_name]}"
                    else:
                        used_names[base_name] = 1
                        folder_name = base_name

                    out = self.output_base / folder_name
                    export_spine(scene, out, editor=self.editor)
                    self.progress.emit(f"[{i + 1}/{total}] 完成: {src.name} -> {folder_name}/")
                success += 1
            except Exception as e:
                self.progress.emit(f"[{i + 1}/{total}] 失败: {src} - {e}")
                fail += 1

        self.finished_signal.emit(success, fail)


class FolderScanWorker(QThread):
    """Background thread: scan a folder recursively, detect valid AssetBundle files,
    and filter out useless data (non-Unity bundles, unrelated formats, etc.)."""
    progress = pyqtSignal(str)
    finished_signal = pyqtSignal(list, int, int)  # valid_paths, skipped_count, error_count

    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder

    def run(self):
        valid: list[Path] = []
        skipped = 0
        errors = 0

        # Collect all regular files recursively
        all_files: list[Path] = []
        try:
            for root, dirs, filenames in os.walk(self.folder):
                # Skip hidden/system directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fn in filenames:
                    if fn.startswith("."):
                        continue
                    all_files.append(Path(root) / fn)
        except Exception as e:
            self.progress.emit(f"扫描文件夹时出错: {e}")
            self.finished_signal.emit([], 0, 0)
            return

        total = len(all_files)
        self.progress.emit(f"发现 {total} 个文件，正在检测可用的 AssetBundle...")

        # Quick validity check for each file
        for i, fpath in enumerate(all_files):
            # Skip known non-bundle file types quickly (no need to try UnityPy)
            ext = fpath.suffix.lower()
            if ext in (".txt", ".log", ".json", ".xml", ".csv", ".meta",
                       ".md", ".ini", ".cfg", ".yaml", ".yml", ".toml",
                       ".py", ".js", ".ts", ".html", ".css", ".bat", ".sh",
                       ".dll", ".so", ".dylib", ".exe", ".lib", ".pdb",
                       ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov",
                       ".zip", ".rar", ".7z", ".tar", ".gz",
                       ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                       ".ttf", ".otf", ".psd", ".ai", ".blend", ".fbx", ".max", ".ma", ".mb",
                       ".asset", ".prefab", ".unity", ".mat",
                       ".cs", ".hlsl", ".cginc", ".shader", ".compute",
                       ".anim", ".controller", ".overrideController",
                       ".physicMaterial", ".physicsMaterial2D",
                       ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".tif", ".gif",
                       ".hdr", ".exr", ".svg",
                       ):
                skipped += 1
                continue

            # Skip directories disguised as files
            if fpath.is_dir():
                skipped += 1
                continue

            # Skip files that are too small (< 1 KB) or too large (> 2 GB)
            try:
                fsize = fpath.stat().st_size
                if fsize < 1024:
                    skipped += 1
                    continue
                if fsize > 2 * 1024 * 1024 * 1024:  # 2 GB
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue

            # Try to open with UnityPy to verify it's a real AssetBundle
            try:
                env = UnityPy.load(str(fpath))
                # Quick check: does it have any recognizable Unity objects?
                obj_count = sum(1 for _ in env.objects)
                if obj_count == 0:
                    skipped += 1
                    continue
                # Check if it contains at least one SkinnedMeshRenderer or SpriteRenderer
                # (i.e., it's a character/sprite bundle, not just a shader/material bundle)
                has_renderer = False
                for o in env.objects:
                    if o.type.name in ("SkinnedMeshRenderer", "SpriteRenderer"):
                        has_renderer = True
                        break
                if not has_renderer:
                    # Still valid if it has Texture2D + Animator (might be a simpler bundle)
                    has_texture = any(o.type.name == "Texture2D" for o in env.objects)
                    has_animator = any(o.type.name == "Animator" for o in env.objects)
                    if has_texture and has_animator:
                        has_renderer = True
                if has_renderer:
                    valid.append(fpath)
                    self.progress.emit(f"  ✓ [{i + 1}/{total}] 有效: {fpath.name}")
                else:
                    skipped += 1
            except Exception:
                # Not a valid AssetBundle or corrupted
                skipped += 1

            # Emit progress periodically
            if (i + 1) % 50 == 0:
                self.progress.emit(f"扫描进度: {i + 1}/{total} (有效: {len(valid)}, 跳过: {skipped})")

        self.progress.emit(
            f"扫描完成: {total} 个文件中, "
            f"找到 {len(valid)} 个有效 AssetBundle, "
            f"跳过 {skipped} 个无用文件"
        )
        self.finished_signal.emit(valid, skipped, errors)


# =============  Preview Canvas  =============

class PreviewCanvas(QOpenGLWidget):
    """OpenGL-accelerated canvas for rendering Spine animation preview.

    Uses GPU-accelerated rendering (OpenGL skinning + rasterization).
    Falls back to QPainter + cv2 CPU rendering if OpenGL is unavailable.

    Wheel events adjust zoom factor for pixel-peeping.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine: Optional[PreviewEngine] = None
        self.playing = False
        self.time = 0.0
        self.speed = 1.0
        self.loop = True
        self._frame_bgra: Optional[np.ndarray] = None
        self._frame_qimage: Optional[QImage] = None
        self._frame_qimage_dirty = True
        self._skeleton_lines: list = []
        self._last_tick = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(16)  # ~60fps UI tick
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # ---- zoom state ----
        self._zoom_factor = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 5.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._pan_orig_x = 0.0
        self._pan_orig_y = 0.0

        # Render state
        self._render_lock = threading.Lock()
        self._render_event = threading.Event()
        self._render_t = 0.0
        self._render_thread: Optional[threading.Thread] = None
        self._render_running = False
        self._use_gpu = HAS_OPENGL
        self._gl_ready = False
        self._last_painted_version: int = -1  # avoid re-rendering same frame
        self._view_changed = True  # force repaint when zoom/pan changes

    def initializeGL(self):
        """Called once when the OpenGL context is ready."""
        if not HAS_OPENGL:
            self._use_gpu = False
            return
        try:
            gl.glClearColor(0.05, 0.05, 0.10, 0.0)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFuncSeparate(
                gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,
                gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA,
            )
            self._gl_ready = True
            self._use_gpu = True

            # If engine already loaded, upload GPU data now
            if self.engine is not None and self.engine.gpu is not None:
                self._init_gpu_renderer()
        except Exception as e:
            print(f"[GL] initializeGL failed: {e}")
            self._use_gpu = False

    def _init_gpu_renderer(self):
        """Initialize GPU renderer and upload geometry on the GL thread."""
        if not self._gl_ready or self.engine is None:
            return
        gpu = self.engine.gpu
        if gpu is None:
            return
        try:
            if not gpu._initialized:
                ok = gpu.init_gl()
                if not ok:
                    self._use_gpu = False
                    return
            gpu.upload_geometry(self.engine)
            self.engine._gpu_upload_pending = False
            print(f"[GL] GPU renderer ready: {len(gpu._part_gpu)} parts, "
                  f"{gpu._gpu_bone_count} bones")
        except Exception as e:
            print(f"[GL] GPU init failed: {e}")
            self._use_gpu = False

    def resizeGL(self, w, h):
        self._view_changed = True
        self._last_painted_version = -1  # force repaint after resize
        if self._gl_ready:
            gl.glViewport(0, 0, w * self.devicePixelRatio(), h * self.devicePixelRatio())

    def set_engine(self, engine: PreviewEngine):
        self._stop_render_thread()
        self.engine = engine
        self.time = 0.0
        self.playing = False
        self._frame_bgra = None
        self._frame_qimage = None
        self._frame_qimage_dirty = True
        self._skeleton_lines = []
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._timer.stop()
        self._last_painted_version = -1  # force repaint for new engine

        # Initialize GPU on GL thread if ready; otherwise initializeGL will handle it
        if self._gl_ready and self.engine.gpu is not None:
            self._init_gpu_renderer()
        elif not self._gl_ready:
            # initializeGL() will be called by Qt when the widget is first shown,
            # and it will call _init_gpu_renderer() there
            pass

        self._request_render(0.0)
        self.update()

    def play(self):
        if not self.engine or not self.engine.current_anim:
            return
        self.playing = True
        self._last_tick = time.time()
        self._ensure_render_thread()
        self._timer.start()

    def pause(self):
        self.playing = False
        self._timer.stop()

    def stop(self):
        self.playing = False
        self._timer.stop()
        self.time = 0.0
        self._request_render(0.0)
        self.update()

    # ------------------------------------------------------------------
    # Background render thread (event-driven)
    # ------------------------------------------------------------------
    def _ensure_render_thread(self):
        if self._render_thread is not None and self._render_running:
            return
        self._render_running = True
        self._render_event.clear()
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def _stop_render_thread(self):
        self._render_running = False
        self._render_event.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=1.0)
            self._render_thread = None

    def _request_render(self, t: float, target_w: Optional[int] = None):
        with self._render_lock:
            self._render_t = t
            if target_w is not None:
                self._render_target_w = target_w
        if self._render_thread is not None and self._render_running:
            self._render_event.set()
        else:
            # Render synchronously when thread isn't running (e.g. initial frame)
            self._do_render()

    def _do_render(self):
        """Perform one render synchronously (used for initial frame or when thread is off)."""
        with self._render_lock:
            t = self._render_t
            target_w = getattr(self, '_render_target_w', None)

        if self.engine is None:
            return

        try:
            if self._use_gpu and self.engine.gpu is not None and self.engine.gpu._initialized:
                self.engine._prepare_gpu_render_data(t)
                with self._render_lock:
                    self._frame_qimage_dirty = True
                    self._skeleton_lines = self.engine._gpu_skeleton_lines
            else:
                frame_bgra = self.engine.render_frame(t, target_w=target_w)
                sk_lines = []
                if self.engine.show_skeleton:
                    sk_lines = self.engine.render_skeleton_overlay(t)
                with self._render_lock:
                    self._frame_bgra = frame_bgra
                    self._frame_qimage_dirty = True
                    self._skeleton_lines = sk_lines
        except Exception as e:
            print(f"[Render] error: {e}")

    def _render_loop(self):
        """Event-driven render loop: compute bone data + opacity, trigger GPU paint."""
        while self._render_running:
            self._render_event.wait()
            if not self._render_running:
                return
            self._render_event.clear()

            with self._render_lock:
                t = self._render_t
                target_w = getattr(self, '_render_target_w', None)

            if self.engine is not None:
                try:
                    # For GPU path: only compute bone data here, rendering happens in paintGL()
                    if self._use_gpu and self.engine.gpu is not None and self.engine.gpu._initialized:
                        # Prepare GPU render data (bone matrices + opacities + skeleton lines)
                        # All derived from a single _interpolate_cached_frame call
                        self.engine._prepare_gpu_render_data(t)
                        with self._render_lock:
                            self._frame_qimage_dirty = True
                            self._skeleton_lines = self.engine._gpu_skeleton_lines
                            self._gpu_render_pending = True
                    else:
                        # CPU fallback: render full frame in this thread
                        frame_bgra = self.engine.render_frame(t, target_w=target_w)
                        sk_lines = []
                        if self.engine.show_skeleton:
                            sk_lines = self.engine.render_skeleton_overlay(t)
                        with self._render_lock:
                            self._frame_bgra = frame_bgra
                            self._frame_qimage_dirty = True
                            self._skeleton_lines = sk_lines
                except Exception as e:
                    print(f"[Render] error: {e}")

    # ------------------------------------------------------------------
    # Qt paint / tick / input
    # ------------------------------------------------------------------
    def _tick(self):
        if not self.engine or not self.engine.current_anim:
            return
        now = time.time()
        dt = (now - self._last_tick) * self.speed
        self._last_tick = now
        self.time += dt

        duration = self.engine.anim_stop
        if duration <= 0:
            duration = 1.0
        if self.time >= duration:
            if self.loop:
                self.time = self.time % duration
            else:
                self.time = duration
                self.playing = False
                self._timer.stop()

        needed_w = self._needed_render_width()
        self._request_render(self.time, target_w=needed_w)
        self.update()

    def _needed_render_width(self) -> int:
        w = self.width()
        if w <= 0:
            return PREVIEW_RENDER_W
        needed = int(w * self._zoom_factor)
        return max(PREVIEW_MIN_RENDER_W, min(PREVIEW_MAX_RENDER_W, needed))

    def _img_display_rect(self, iw: int, ih: int) -> tuple:
        w, h = self.width(), self.height()
        fit_scale = min(w / iw, h / ih) if iw > 0 and ih > 0 else 1.0
        scale = fit_scale * self._zoom_factor
        dw = int(iw * scale)
        dh = int(ih * scale)
        dx = (w - dw) // 2 + int(self._pan_x)
        dy = (h - dh) // 2 + int(self._pan_y)
        return dx, dy, dw, dh, scale

    def paintGL(self):
        """OpenGL paint: render GPU-accelerated Spine mesh directly to the widget.

        All uniform locations are cached in the GPURenderer at init time,
        so this path avoids the expensive glGetUniformLocation per-part calls.

        Uses a frame version counter to skip re-rendering when the bone data
        hasn't changed (render thread produces frames slower than display refresh).
        """
        if not self._gl_ready or not self.engine or not self._use_gpu:
            # CPU fallback mode: this must be a complete no-op. Any GL calls
            # here (glClear, glDisable(GL_BLEND), ...) run on the same FBO
            # that Qt's own QPainter compositing uses for paintEvent()'s
            # drawImage() call right after this - leftover GL state from an
            # unconditional glClear/glDisable(GL_BLEND) here can stop that
            # image from showing up at all, producing a black canvas even
            # though a valid frame was rendered and handed to QPainter.
            return

        # Skip if no new frame data since last paint AND view hasn't changed
        frame_version = self.engine._gpu_frame_version
        if frame_version == self._last_painted_version and not self._view_changed:
            return
        self._last_painted_version = frame_version
        self._view_changed = False

        w, h = self.width(), self.height()
        dpr = self.devicePixelRatio()
        pw, ph = int(w * dpr), int(h * dpr)

        gl.glViewport(0, 0, pw, ph)
        gl.glClearColor(0.05, 0.05, 0.10, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(
            gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,
            gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA,
        )

        gpu = self.engine.gpu
        if gpu is None or not gpu._initialized or not gpu._part_gpu:
            gl.glDisable(gl.GL_BLEND)
            return

        # Get prepared bone data from engine (set by _prepare_gpu_render_data)
        bone_mats = getattr(self.engine, '_gpu_bone_mats', None)
        opacities = getattr(self.engine, '_gpu_opacities', None)
        if bone_mats is None or opacities is None:
            gl.glDisable(gl.GL_BLEND)
            return

        try:
            gl.glUseProgram(gpu._shader_prog)
            gl.glBindVertexArray(gpu._vao)

            # Build ortho projection with zoom/pan (same logic as before)
            engine = self.engine
            minx, miny = engine.minx, engine.miny
            maxx, maxy = engine.maxx, engine.maxy
            ww = maxx - minx
            wh = maxy - miny
            if ww < 1e-6 or wh < 1e-6:
                gl.glBindVertexArray(0)
                gl.glUseProgram(0)
                gl.glDisable(gl.GL_BLEND)
                return

            zoom = self._zoom_factor
            pan_x = self._pan_x
            pan_y = self._pan_y
            widget_w, widget_h = self.width(), self.height()
            fit_scale = min(widget_w / ww, widget_h / wh) if widget_w > 0 and wh > 0 else 1.0
            total_scale = fit_scale * zoom
            pan_wx = pan_x / total_scale if abs(total_scale) > 1e-6 else 0
            pan_wy = -pan_y / total_scale if abs(total_scale) > 1e-6 else 0

            half_w = widget_w / (2.0 * total_scale) if abs(total_scale) > 1e-6 else ww / 2.0
            half_h = widget_h / (2.0 * total_scale) if abs(total_scale) > 1e-6 else wh / 2.0
            center_x = (minx + maxx) / 2.0 + pan_wx
            center_y = (miny + maxy) / 2.0 + pan_wy

            left = center_x - half_w
            right = center_x + half_w
            bottom = center_y - half_h
            top = center_y + half_h

            proj = np.array([
                [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
                [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

            # Use cached uniform locations (NO glGetUniformLocation calls!)
            gl.glUniformMatrix4fv(gpu._loc_uMVP, 1, gl.GL_TRUE, proj.T.copy())

            # Upload bone matrices – cached location
            bone_flat = np.ascontiguousarray(
                bone_mats.transpose(0, 2, 1).reshape(-1), dtype=np.float32,
            )
            gl.glUniformMatrix4fv(gpu._loc_uBoneMats, len(bone_mats), gl.GL_TRUE, bone_flat)

            # Pre-compute stride & attribute offsets once
            stride = 12 * 4
            c_void_0 = c_void_p(0)
            c_void_8 = c_void_p(8)
            c_void_16 = c_void_p(16)
            c_void_32 = c_void_p(32)

            # Draw each part
            for i, pg in enumerate(gpu._part_gpu):
                if i >= len(opacities):
                    opacity = 1.0
                else:
                    opacity = max(0.0, min(1.0, opacities[i]))
                if opacity < 0.001:
                    continue

                gl.glUniform1f(gpu._loc_uOpacity, opacity)

                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, pg["texture"])
                gl.glUniform1i(gpu._loc_uTexture, 0)

                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, pg["vbo"])
                gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_0)
                gl.glEnableVertexAttribArray(0)
                gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_8)
                gl.glEnableVertexAttribArray(1)
                gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_16)
                gl.glEnableVertexAttribArray(2)
                gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_32)
                gl.glEnableVertexAttribArray(3)

                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, pg["ebo"])
                gl.glDrawElements(gl.GL_TRIANGLES, pg["index_count"], gl.GL_UNSIGNED_INT, c_void_0)

            gl.glDisableVertexAttribArray(0)
            gl.glDisableVertexAttribArray(1)
            gl.glDisableVertexAttribArray(2)
            gl.glDisableVertexAttribArray(3)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)

        except Exception as e:
            print(f"[GL] paintGL error: {e}")
        finally:
            gl.glDisable(gl.GL_BLEND)

    def paintEvent(self, event):
        """Override paintEvent to draw skeleton overlay + UI text on top of OpenGL."""
        # First call OpenGL paint (paintGL is called automatically)
        super().paintEvent(event)

        # Then overlay with QPainter for skeleton lines and text
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        if not self.engine:
            painter.setPen(QColor(128, 128, 144))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "请先加载 AssetBundle")
            painter.end()
            return

        # Get skeleton lines
        with self._render_lock:
            sk_lines = list(self._skeleton_lines) if self._skeleton_lines else []

        # Draw skeleton overlay (projected from world coords to widget coords)
        if sk_lines and self.engine and self.engine.show_skeleton and self._use_gpu:
            engine = self.engine
            minx, miny = engine.minx, engine.miny
            maxx, maxy = engine.maxx, engine.maxy
            ww = maxx - minx
            wh = maxy - miny
            if ww > 1e-6 and wh > 1e-6:
                zoom = self._zoom_factor
                fit_scale = min(w / ww, h / wh) if w > 0 else 1.0
                total_scale = fit_scale * zoom
                pan_wx = self._pan_x / total_scale if abs(total_scale) > 1e-6 else 0
                pan_wy = -self._pan_y / total_scale if abs(total_scale) > 1e-6 else 0
                center_x = (minx + maxx) / 2.0 + pan_wx
                center_y = (miny + maxy) / 2.0 + pan_wy
                half_w = w / (2.0 * total_scale) if abs(total_scale) > 1e-6 else ww / 2.0
                half_h = h / (2.0 * total_scale) if abs(total_scale) > 1e-6 else wh / 2.0

                def world_to_widget(wx, wy):
                    sx = (wx - (center_x - half_w)) / (2.0 * half_w) * w
                    sy = h - (wy - (center_y - half_h)) / (2.0 * half_h) * h
                    return sx, sy

                painter.setPen(QPen(QColor(233, 69, 96, 180), 1.5))
                painter.setBrush(QBrush(QColor(233, 69, 96)))
                for (px, py, cx, cy) in sk_lines:
                    spx, spy = world_to_widget(px, py)
                    scx, scy = world_to_widget(cx, cy)
                    painter.drawLine(int(spx), int(spy), int(scx), int(scy))
                    painter.drawEllipse(int(scx) - 3, int(scy) - 3, 6, 6)
        elif self.engine and not self._use_gpu:
            # CPU path: nothing else draws the frame (no custom GL calls run
            # when GPU init failed), so the rendered image itself has to be
            # drawn here too - it must NOT be gated on sk_lines/show_skeleton,
            # those only control the optional bone overlay drawn on top of it.
            with self._render_lock:
                frame_bgra = self._frame_bgra
            if frame_bgra is not None and frame_bgra.size > 0:
                if self._frame_qimage_dirty:
                    rgba = np.ascontiguousarray(frame_bgra[:, :, [2, 1, 0, 3]])
                    ih_img, iw_img = rgba.shape[:2]
                    self._frame_qimage = QImage(
                        rgba.data, iw_img, ih_img, iw_img * 4, QImage.Format_RGBA8888,
                    ).copy()
                    self._frame_qimage_dirty = False

                if self._frame_qimage is not None and not self._frame_qimage.isNull():
                    iw_img = self._frame_qimage.width()
                    ih_img = self._frame_qimage.height()
                    dx, dy, dw, dh, scale = self._img_display_rect(iw_img, ih_img)
                    if dw > 0 and dh > 0:
                        painter.drawImage(QRectF(dx, dy, dw, dh), self._frame_qimage)
                    if sk_lines and self.engine.show_skeleton:
                        scale_x = dw / iw_img
                        scale_y = dh / ih_img
                        painter.setPen(QPen(QColor(233, 69, 96, 180), 1.5))
                        painter.setBrush(QBrush(QColor(233, 69, 96)))
                        for (px, py, cx, cy) in sk_lines:
                            spx = px * scale_x + dx
                            spy = py * scale_y + dy
                            scx = cx * scale_x + dx
                            scy = cy * scale_y + dy
                            painter.drawLine(int(spx), int(spy), int(scx), int(scy))
                            painter.drawEllipse(int(scx) - 3, int(scy) - 3, 6, 6)

        # Overlay text
        if self.engine and self.engine.current_anim:
            painter.setPen(QColor(200, 200, 220))
            painter.setFont(QFont("Consolas", 11))
            duration = self.engine.anim_stop
            text = f"{self.engine.current_anim}  |  {self.time:.2f}s / {duration:.2f}s"
            if self.playing:
                text += "  ▶"
            if abs(self._zoom_factor - 1.0) > 0.01:
                text += f"  {self._zoom_factor:.1f}x"
            if self._use_gpu:
                text += "  [GPU]"
            painter.drawText(10, 24, text)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse wheel zoom
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        if not self.engine:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_step = 0.1 if abs(delta) < 120 else 0.2
        old_zoom = self._zoom_factor
        if delta > 0:
            self._zoom_factor = min(self._max_zoom, self._zoom_factor + zoom_step)
        else:
            self._zoom_factor = max(self._min_zoom, self._zoom_factor - zoom_step)

        if abs(self._zoom_factor - old_zoom) < 0.001:
            return

        ratio = self._zoom_factor / old_zoom
        mx, my = event.pos().x(), event.pos().y()
        w, h = self.width(), self.height()
        center_x, center_y = w / 2.0, h / 2.0
        self._pan_x = int(mx - ratio * (mx - center_x - self._pan_x) - center_x)
        self._pan_y = int(my - ratio * (my - center_y - self._pan_y) - center_y)

        self._view_changed = True
        if self.engine and self.engine.current_anim:
            self._request_render(self.time, target_w=self._needed_render_width())
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._zoom_factor > 1.01
        ):
            self._panning = True
            self._pan_start_x = event.pos().x()
            self._pan_start_y = event.pos().y()
            self._pan_orig_x = self._pan_x
            self._pan_orig_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            dx = event.pos().x() - self._pan_start_x
            dy = event.pos().y() - self._pan_start_y
            self._pan_x = self._pan_orig_x + dx
            self._pan_y = self._pan_orig_y + dy
            self._view_changed = True
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# =============  Main Window  =============

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Unity AssetBundle → Spine {SPINE_VERSION} 转换工具")
        self.setMinimumSize(1200, 750)
        self.setAcceptDrops(True)  # Enable drag-and-drop

        # State
        self.current_src: Optional[Path] = None
        self.preview_engine: Optional[PreviewEngine] = None
        self.batch_sources: list[Path] = []
        self.convert_worker: Optional[ConvertWorker] = None
        self.preview_worker: Optional[PreviewOnlyWorker] = None
        self.batch_worker: Optional[BatchConvertWorker] = None
        self.folder_scan_worker: Optional[FolderScanWorker] = None
        self.scanned_folder: Optional[Path] = None  # source folder for batch export
        self.folder_output_dir: Optional[Path] = None  # target folder for batch export
        self._batch_list_updating = False  # guard to avoid itemChanged recursion
        self._batch_rename_map: dict[str, str] = {}  # original stem -> custom name

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ---- Left panel ----
        left_panel = QWidget()
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # File selection
        grp_file = QGroupBox("📁 源文件")
        fl = QVBoxLayout(grp_file)
        fl.setSpacing(4)

        fl_sel = QHBoxLayout()
        self.lbl_src = QLabel("未选择文件")
        self.lbl_src.setWordWrap(True)
        self.lbl_src.setStyleSheet("color:#808090; font-size:12px;")
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_src)
        fl_sel.addWidget(self.lbl_src, 1)
        fl_sel.addWidget(self.btn_browse)
        fl.addLayout(fl_sel)

        self.btn_batch_add = QPushButton("📦 添加批量文件...")
        self.btn_batch_add.clicked.connect(self._browse_batch)
        fl.addWidget(self.btn_batch_add)

        self.btn_folder_scan = QPushButton("📁 从文件夹智能导入...")
        self.btn_folder_scan.setMinimumHeight(34)
        self.btn_folder_scan.setStyleSheet(
            "QPushButton { background:#1a4a3e; color:#e0e0e0; font-size:12px; "
            "border-radius:6px; } QPushButton:hover { background:#2a6a5e; } "
            "QPushButton:disabled { background:#222; color:#555; }"
        )
        self.btn_folder_scan.clicked.connect(self._browse_folder_scan)
        fl.addWidget(self.btn_folder_scan)

        self.lbl_batch_count = QLabel("批量: 0 个文件")
        self.lbl_batch_count.setStyleSheet("color:#606070; font-size:11px;")
        fl.addWidget(self.lbl_batch_count)

        # ---- Batch file list with checkboxes ----
        fl_list_toolbar = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.setMaximumWidth(50)
        self.btn_select_all.clicked.connect(lambda: self._batch_list_toggle_all(True))
        fl_list_toolbar.addWidget(self.btn_select_all)
        self.btn_deselect_all = QPushButton("取消全选")
        self.btn_deselect_all.setMaximumWidth(70)
        self.btn_deselect_all.clicked.connect(lambda: self._batch_list_toggle_all(False))
        fl_list_toolbar.addWidget(self.btn_deselect_all)
        fl_list_toolbar.addStretch()
        self.lbl_selected_count = QLabel("已选: 0")
        self.lbl_selected_count.setStyleSheet("color:#808090; font-size:10px;")
        fl_list_toolbar.addWidget(self.lbl_selected_count)
        fl.addLayout(fl_list_toolbar)

        self.batch_file_list = QListWidget()
        self.batch_file_list.setMaximumHeight(180)
        self.batch_file_list.setStyleSheet(
            "QListWidget { background:#0d0d1a; border:1px solid #2a2a4e; border-radius:4px; "
            "color:#c0c0d0; font-size:11px; }"
            "QListWidget::item { padding:2px 4px; }"
            "QListWidget::item:hover { background:#1a2a4e; }"
            "QListWidget::item:selected { background:#0f3460; }"
        )
        self.batch_file_list.itemChanged.connect(self._on_batch_item_changed)
        self.batch_file_list.itemDoubleClicked.connect(self._on_batch_item_double_clicked)
        self.batch_file_list.setVisible(False)
        fl.addWidget(self.batch_file_list)

        # Rename toolbar (appears when files are loaded)
        self.batch_rename_bar = QWidget()
        br_layout = QHBoxLayout(self.batch_rename_bar)
        br_layout.setContentsMargins(0, 2, 0, 2)
        br_layout.addWidget(QLabel("导出前缀:"))
        self.txt_export_prefix = QLineEdit()
        self.txt_export_prefix.setPlaceholderText("留空使用原名")
        self.txt_export_prefix.setStyleSheet(
            "QLineEdit { background:#1a1a3e; border:1px solid #2a2a4e; border-radius:4px; "
            "padding:3px 6px; font-size:11px; color:#e0e0e0; }"
        )
        br_layout.addWidget(self.txt_export_prefix, 1)
        self.chk_use_stem_as_name = QCheckBox("自动去扩展名")
        self.chk_use_stem_as_name.setChecked(True)
        self.chk_use_stem_as_name.setStyleSheet("font-size:10px;")
        br_layout.addWidget(self.chk_use_stem_as_name)
        self.batch_rename_bar.setVisible(False)
        fl.addWidget(self.batch_rename_bar)

        grp_file.setLayout(fl)
        left_layout.addWidget(grp_file)

        # Output settings
        grp_out = QGroupBox("⚙ 导出设置")
        ol = QVBoxLayout(grp_out)
        ol.setSpacing(4)

        ol_mode = QHBoxLayout()
        ol_mode.addWidget(QLabel("模式:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Spine Editor (editor)", "Runtime (runtime)"])
        ol_mode.addWidget(self.cmb_mode, 1)
        ol.addLayout(ol_mode)

        ol_dir = QHBoxLayout()
        self.lbl_output = QLabel("默认 (同目录)")
        self.lbl_output.setStyleSheet("color:#808090; font-size:12px;")
        self.lbl_output.setWordWrap(True)
        self.btn_out_dir = QPushButton("输出目录...")
        self.btn_out_dir.clicked.connect(self._browse_output)
        ol_dir.addWidget(self.lbl_output, 1)
        ol_dir.addWidget(self.btn_out_dir)
        ol.addLayout(ol_dir)

        # Scale
        ol_scale = QHBoxLayout()
        ol_scale.addWidget(QLabel("画布宽度:"))
        self.spin_target_w = QSpinBox()
        self.spin_target_w.setRange(200, 4096)
        self.spin_target_w.setValue(TARGET_W)
        self.spin_target_w.setSingleStep(100)
        self.spin_target_w.setToolTip("目标画布宽度 (Spine scale)")
        ol_scale.addWidget(self.spin_target_w, 1)
        ol.addLayout(ol_scale)

        grp_out.setLayout(ol)
        left_layout.addWidget(grp_out)

        # Preview only button (no Spine export)
        self.btn_preview = QPushButton("👁 仅预览 (不导出)")
        self.btn_preview.setMinimumHeight(34)
        self.btn_preview.setStyleSheet(
            "QPushButton { background:#0f3460; color:#e0e0e0; font-size:12px; "
            "border-radius:6px; } QPushButton:hover { background:#1a4a80; } "
            "QPushButton:disabled { background:#222; color:#555; }"
        )
        self.btn_preview.clicked.connect(self._start_preview)
        left_layout.addWidget(self.btn_preview)

        # Convert button
        btn_row = QHBoxLayout()
        self.btn_convert = QPushButton("▶ 转换并预览")
        self.btn_convert.setMinimumHeight(38)
        self.btn_convert.setStyleSheet(
            "QPushButton { background:#e94560; color:#fff; font-size:14px; font-weight:bold; "
            "border-radius:6px; } QPushButton:hover { background:#ff6b81; } "
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self.btn_convert.clicked.connect(self._start_convert)
        btn_row.addWidget(self.btn_convert)
        left_layout.addLayout(btn_row)

        self.btn_batch_convert = QPushButton("📦 批量导出")
        self.btn_batch_convert.setMinimumHeight(34)
        self.btn_batch_convert.clicked.connect(self._start_batch)
        self.btn_batch_convert.setEnabled(False)
        left_layout.addWidget(self.btn_batch_convert)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Log
        grp_log = QGroupBox("📋 日志")
        ll = QVBoxLayout(grp_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(150)
        self.txt_log.setStyleSheet("QTextEdit { background:#0d0d1a; color:#a0a0b0; font-size:11px; }")
        ll.addWidget(self.txt_log)
        grp_log.setLayout(ll)
        left_layout.addWidget(grp_log)

        left_layout.addStretch()

        # ---- Right panel (Preview) ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Preview toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(36)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_play.setToolTip("播放/暂停 (空格键)")
        toolbar.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(36)
        self.btn_stop.clicked.connect(self._stop_anim)
        self.btn_stop.setToolTip("停止")
        toolbar.addWidget(self.btn_stop)

        self.btn_loop = QPushButton("🔁")
        self.btn_loop.setFixedWidth(36)
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.clicked.connect(self._toggle_loop)
        self.btn_loop.setToolTip("循环播放")
        toolbar.addWidget(self.btn_loop)

        toolbar.addWidget(QLabel("速度:"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 5.0)
        self.spin_speed.setValue(1.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setFixedWidth(60)
        self.spin_speed.valueChanged.connect(self._speed_changed)
        toolbar.addWidget(self.spin_speed)

        toolbar.addWidget(QLabel("动画:"))
        self.cmb_anim = QComboBox()
        self.cmb_anim.setMinimumWidth(200)
        self.cmb_anim.currentTextChanged.connect(self._anim_changed)
        toolbar.addWidget(self.cmb_anim, 1)

        # Time slider
        self.sld_time = QSlider(Qt.Horizontal)
        self.sld_time.setRange(0, 1000)
        self.sld_time.setValue(0)
        self.sld_time.sliderPressed.connect(self._slider_pressed)
        self.sld_time.sliderReleased.connect(self._slider_released)
        self.sld_time.valueChanged.connect(self._slider_moved)
        toolbar.addWidget(self.sld_time, 2)

        self.lbl_time = QLabel("0.00s")
        self.lbl_time.setFixedWidth(55)
        self.lbl_time.setStyleSheet("color:#e94560; font-family:Consolas; font-size:13px;")
        toolbar.addWidget(self.lbl_time)

        self.chk_skeleton = QCheckBox("骨骼")
        self.chk_skeleton.setChecked(True)
        self.chk_skeleton.toggled.connect(self._toggle_skeleton)
        toolbar.addWidget(self.chk_skeleton)

        right_layout.addLayout(toolbar)

        # Canvas
        self.canvas = PreviewCanvas()
        right_layout.addWidget(self.canvas, 1)

        # Info bar
        info_bar = QHBoxLayout()
        self.lbl_info = QLabel("就绪")
        self.lbl_info.setStyleSheet("color:#808090; font-size:11px;")
        info_bar.addWidget(self.lbl_info)
        info_bar.addStretch()
        self.lbl_fps = QLabel("")
        self.lbl_fps.setStyleSheet("color:#606070; font-size:11px;")
        info_bar.addWidget(self.lbl_fps)
        right_layout.addLayout(info_bar)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([340, 860])
        main_layout.addWidget(splitter)

        # Keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._toggle_play()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        """Accept file drag events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # Visual feedback: highlight border
            self.centralWidget().setStyleSheet(
                "QWidget { border: 2px solid #e94560; }"
            )

    def dragLeaveEvent(self, event):
        """Reset visual feedback when drag leaves."""
        self.centralWidget().setStyleSheet("")

    def dropEvent(self, event):
        """Handle dropped files/folders: load single file or scan folder."""
        self.centralWidget().setStyleSheet("")
        if not event.mimeData().hasUrls():
            return
        urls = event.mimeData().urls()
        if not urls:
            return

        path = Path(urls[0].toLocalFile())
        if not path.exists():
            self._log(f"路径不存在: {path}")
            return

        # If a folder is dropped, auto-scan it
        if path.is_dir():
            self._log(f"拖入文件夹: {path}")
            # Ask for output directory
            out_folder = QFileDialog.getExistingDirectory(
                self, "选择输出目标文件夹",
                str(path.parent / "spine_export"),
            )
            if not out_folder:
                return
            self.scanned_folder = path
            self.folder_output_dir = Path(out_folder)
            self._log(f"输出目标文件夹: {self.folder_output_dir}")

            self._disable_all_buttons()
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)

            self.folder_scan_worker = FolderScanWorker(self.scanned_folder)
            self.folder_scan_worker.progress.connect(self._log)
            self.folder_scan_worker.finished_signal.connect(self._on_folder_scan_done)
            self.folder_scan_worker.start()
            return

        # Single file
        self.current_src = path
        self.lbl_src.setText(path.name)
        self.lbl_src.setStyleSheet("color:#e0e0e0; font-size:12px;")
        self._log(f"拖入文件: {path}")

        # If multiple files dropped, add all to batch
        if len(urls) > 1:
            self.batch_sources = [Path(u.toLocalFile()) for u in urls if Path(u.toLocalFile()).exists()]
            self._batch_rename_map.clear()
            self._populate_batch_file_list(self.batch_sources)
            self.lbl_batch_count.setText(f"批量: {len(self.batch_sources)} 个文件")
            self.btn_batch_convert.setEnabled(True)
            self.batch_file_list.setVisible(True)
            self.batch_rename_bar.setVisible(True)
            self.btn_select_all.setVisible(True)
            self.btn_deselect_all.setVisible(True)
            self.lbl_selected_count.setVisible(True)
            self._update_selected_count()
            self._log(f"批量添加 {len(self.batch_sources)} 个文件")
        else:
            # Auto-preview the single dropped file
            self._start_preview()

    # ---- Style ----

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QWidget { color: #e0e0e0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
            QGroupBox {
                font-size: 13px; font-weight: bold; color: #a0a0b0;
                border: 1px solid #2a2a4e; border-radius: 8px;
                margin-top: 12px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px;
                color: #c0c0d0;
            }
            QPushButton {
                background: #1e2a4e; border: 1px solid #2a3a5e; border-radius: 5px;
                padding: 6px 12px; color: #e0e0e0; font-size: 12px;
            }
            QPushButton:hover { background: #2a3a5e; border-color: #3a4a6e; }
            QPushButton:pressed { background: #0f1a3e; }
            QPushButton:checked { background: #e94560; border-color: #e94560; color: #fff; }
            QPushButton:disabled { background: #222; color: #555; border-color: #333; }
            QComboBox {
                background: #1a1a3e; border: 1px solid #2a2a4e; border-radius: 4px;
                padding: 4px 8px; font-size: 12px; min-height: 20px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1a1a3e; color: #e0e0e0; selection-background-color: #0f3460;
            }
            QSpinBox, QDoubleSpinBox {
                background: #1a1a3e; border: 1px solid #2a2a4e; border-radius: 4px;
                padding: 3px 6px; font-size: 12px;
            }
            QSlider::groove:horizontal {
                height: 6px; background: #2a2a4e; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -5px 0;
                background: #e94560; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #e94560; border-radius: 3px; }
            QProgressBar {
                border: 1px solid #2a2a4e; border-radius: 4px; text-align: center;
                background: #0d0d1a; font-size: 11px; height: 16px;
            }
            QProgressBar::chunk { background: #e94560; border-radius: 3px; }
            QCheckBox { spacing: 6px; font-size: 12px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QScrollArea { border: none; }
        """)

    # ---- Slots ----

    def _browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Unity AssetBundle", "",
            "AssetBundle (__data *);;所有文件 (*)"
        )
        if path:
            self.current_src = Path(path)
            self.lbl_src.setText(self.current_src.name)
            self.lbl_src.setStyleSheet("color:#e0e0e0; font-size:12px;")
            self._log(f"选择文件: {self.current_src}")

    def _browse_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个 AssetBundle 文件", "",
            "AssetBundle (__data *);;所有文件 (*)"
        )
        if paths:
            self.batch_sources = [Path(p) for p in paths]
            self._batch_rename_map.clear()
            self._populate_batch_file_list(self.batch_sources)
            self.lbl_batch_count.setText(f"批量: {len(self.batch_sources)} 个文件")
            self.btn_batch_convert.setEnabled(True)
            self.batch_file_list.setVisible(True)
            self.batch_rename_bar.setVisible(True)
            self.btn_select_all.setVisible(True)
            self.btn_deselect_all.setVisible(True)
            self.lbl_selected_count.setVisible(True)
            self._update_selected_count()
            self._log(f"添加 {len(self.batch_sources)} 个批量文件")

    def _browse_folder_scan(self):
        """Scan a folder recursively, auto-detect valid AssetBundles,
        filter out useless data, then batch export to another folder."""
        folder = QFileDialog.getExistingDirectory(self, "选择包含 AssetBundle 的文件夹")
        if not folder:
            return

        self.scanned_folder = Path(folder)
        self._log(f"开始扫描文件夹: {self.scanned_folder}")

        # Ask for output directory
        out_folder = QFileDialog.getExistingDirectory(
            self, "选择输出目标文件夹",
            str(self.scanned_folder.parent / "spine_export"),
        )
        if not out_folder:
            return

        self.folder_output_dir = Path(out_folder)
        self._log(f"输出目标文件夹: {self.folder_output_dir}")

        self._disable_all_buttons()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate during scan

        self.folder_scan_worker = FolderScanWorker(self.scanned_folder)
        self.folder_scan_worker.progress.connect(self._log)
        self.folder_scan_worker.finished_signal.connect(self._on_folder_scan_done)
        self.folder_scan_worker.start()

    def _on_folder_scan_done(self, valid_paths: list, skipped: int, errors: int):
        """Called when folder scanning completes. Populates the file list for user review."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        if not valid_paths:
            self._enable_all_buttons()
            self._log("❌ 未找到任何有效的 AssetBundle 文件")
            QMessageBox.warning(self, "扫描结果", "在所选文件夹中未找到任何有效的 AssetBundle 文件。\n\n请确认文件夹中包含 Unity AssetBundle 数据（包含 SkinnedMeshRenderer 或 SpriteRenderer）。")
            return

        self._log(f"✅ 找到 {len(valid_paths)} 个有效文件，跳过 {skipped} 个无用文件")

        # Store as batch sources
        self.batch_sources = valid_paths
        self._batch_rename_map.clear()

        # Populate the checkable file list
        self._populate_batch_file_list(valid_paths)

        self.lbl_batch_count.setText(f"批量: {len(valid_paths)} 个文件 (跳过 {skipped})")
        self.btn_batch_convert.setEnabled(True)
        self.batch_file_list.setVisible(True)
        self.batch_rename_bar.setVisible(True)
        self.btn_select_all.setVisible(True)
        self.btn_deselect_all.setVisible(True)
        self.lbl_selected_count.setVisible(True)

        self._update_selected_count()
        self._enable_all_buttons()

    def _populate_batch_file_list(self, paths: list[Path]):
        """Fill the batch file list with checkable items."""
        self._batch_list_updating = True
        self.batch_file_list.clear()
        for p in paths:
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))  # store full path
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.batch_file_list.addItem(item)
        self._batch_list_updating = False

    def _on_batch_item_changed(self, item: QListWidgetItem):
        """Called when a checkbox is toggled in the batch file list."""
        if self._batch_list_updating:
            return
        self._update_selected_count()

    def _on_batch_item_double_clicked(self, item: QListWidgetItem):
        """Double-click a file in the batch list to preview it."""
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return
        src = Path(path_str)
        if not src.exists():
            QMessageBox.warning(self, "错误", f"文件不存在: {src}")
            return
        self.current_src = src
        self.lbl_src.setText(src.name)
        self.lbl_src.setStyleSheet("color:#e0e0e0; font-size:12px;")
        self._log(f"预览: {src.name}")
        self._start_preview()

    def _batch_list_toggle_all(self, checked: bool):
        """Toggle all items in the batch file list."""
        self._batch_list_updating = True
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.batch_file_list.count()):
            self.batch_file_list.item(i).setCheckState(state)
        self._batch_list_updating = False
        self._update_selected_count()

    def _update_selected_count(self):
        """Update the selected count label."""
        count = 0
        for i in range(self.batch_file_list.count()):
            if self.batch_file_list.item(i).checkState() == Qt.Checked:
                count += 1
        self.lbl_selected_count.setText(f"已选: {count}")

    def _get_checked_paths(self) -> list[Path]:
        """Get all checked file paths from the batch list."""
        result = []
        for i in range(self.batch_file_list.count()):
            item = self.batch_file_list.item(i)
            if item.checkState() == Qt.Checked:
                path_str = item.data(Qt.UserRole)
                if path_str:
                    result.append(Path(path_str))
        return result

    def _get_export_name(self, src: Path) -> str:
        """Determine the export folder name for a source file.
        Uses the rename map if set, otherwise falls back to stem."""
        stem = src.stem if src.suffix else src.name
        if stem in self._batch_rename_map:
            return self._batch_rename_map[stem]
        prefix = self.txt_export_prefix.text().strip()
        if prefix:
            return f"{prefix}_{stem}"
        if self.chk_use_stem_as_name.isChecked():
            return stem
        return src.name

    def _start_folder_batch_export(self):
        """Start batch exporting selected files from the batch list to output directory."""
        # Use only checked files
        checked = self._get_checked_paths()
        if not checked:
            QMessageBox.warning(self, "提示", "没有勾选任何文件，请先在列表中勾选要导出的文件。")
            return

        if not self.folder_output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出目录")
            return

        editor = self.cmb_mode.currentIndex() == 0
        out_base = self.folder_output_dir
        out_base.mkdir(parents=True, exist_ok=True)

        # Set TARGET_W
        import unity_to_spine
        unity_to_spine.TARGET_W = self.spin_target_w.value()

        self._disable_all_buttons()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(checked))
        self.progress_bar.setValue(0)
        self._log(f"开始批量导出 {len(checked)} 个文件...")

        self.batch_worker = BatchConvertWorker(checked, out_base, editor)
        self.batch_worker.progress.connect(self._log)
        self.batch_worker.file_progress.connect(self._on_batch_progress)
        self.batch_worker.finished_signal.connect(self._on_batch_done)
        self.batch_worker.start()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir = Path(path)
            self.lbl_output.setText(str(self.output_dir))
            self.lbl_output.setStyleSheet("color:#e0e0e0; font-size:12px;")
            self._log(f"输出目录: {self.output_dir}")
        else:
            self.output_dir = None
            self.lbl_output.setText("默认 (同目录)")
            self.lbl_output.setStyleSheet("color:#808090; font-size:12px;")

    @property
    def output_dir(self) -> Optional[Path]:
        return getattr(self, "_output_dir", None)

    @output_dir.setter
    def output_dir(self, v):
        self._output_dir = v

    def _start_preview(self):
        """Preview only: load Unity Bundle and render directly, no Spine export."""
        if not self.current_src:
            QMessageBox.warning(self, "提示", "请先选择源文件")
            return

        if not self.current_src.exists():
            QMessageBox.critical(self, "错误", f"文件不存在: {self.current_src}")
            return

        self._disable_all_buttons()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.txt_log.clear()
        self._log("正在加载 Bundle 用于预览...")

        self.preview_worker = PreviewOnlyWorker(self.current_src)
        self.preview_worker.progress.connect(self._log)
        self.preview_worker.finished_signal.connect(self._on_preview_done)
        self.preview_worker.preview_ready.connect(self._on_preview_ready)
        self.preview_worker.start()

    def _on_preview_done(self, success: bool, msg: str):
        self._enable_all_buttons()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        if success:
            self._log(f"✅ {msg}")
            self.lbl_info.setText("预览就绪 (未导出)")
        else:
            self._log(f"❌ {msg}")
            self.lbl_info.setText("预览加载失败")
            QMessageBox.critical(self, "预览失败", msg)

    def _disable_all_buttons(self):
        self.btn_preview.setEnabled(False)
        self.btn_convert.setEnabled(False)
        self.btn_batch_convert.setEnabled(False)
        self.btn_folder_scan.setEnabled(False)
        self.btn_select_all.setEnabled(False)
        self.btn_deselect_all.setEnabled(False)

    def _enable_all_buttons(self):
        self.btn_preview.setEnabled(True)
        self.btn_convert.setEnabled(True)
        self.btn_batch_convert.setEnabled(len(self.batch_sources) > 0)
        self.btn_folder_scan.setEnabled(True)
        self.btn_select_all.setEnabled(True)
        self.btn_deselect_all.setEnabled(True)

    def _start_convert(self):
        if not self.current_src:
            QMessageBox.warning(self, "提示", "请先选择源文件")
            return

        if not self.current_src.exists():
            QMessageBox.critical(self, "错误", f"文件不存在: {self.current_src}")
            return

        editor = self.cmb_mode.currentIndex() == 0
        out = self.output_dir

        self._disable_all_buttons()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.txt_log.clear()
        self._log("开始转换...")

        # Set TARGET_W from spinbox
        import unity_to_spine
        unity_to_spine.TARGET_W = self.spin_target_w.value()

        self.convert_worker = ConvertWorker(self.current_src, out, editor)
        self.convert_worker.progress.connect(self._log)
        self.convert_worker.finished_signal.connect(self._on_convert_done)
        self.convert_worker.preview_ready.connect(self._on_preview_ready)
        self.convert_worker.start()

    def _on_convert_done(self, success: bool, msg: str):
        self._enable_all_buttons()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        if success:
            self._log(f"✅ {msg}")
            self.lbl_info.setText("转换完成")
        else:
            self._log(f"❌ {msg}")
            self.lbl_info.setText("转换失败")
            QMessageBox.critical(self, "转换失败", msg)

    def _on_preview_ready(self, engine: PreviewEngine):
        self.preview_engine = engine
        self.canvas.set_engine(engine)

        # Populate animation list
        self.cmb_anim.blockSignals(True)
        self.cmb_anim.clear()
        for name in engine.animations.keys():
            self.cmb_anim.addItem(name)
        if engine.current_anim:
            idx = self.cmb_anim.findText(engine.current_anim)
            if idx >= 0:
                self.cmb_anim.setCurrentIndex(idx)
        self.cmb_anim.blockSignals(False)

        self.lbl_info.setText(f"骨骼:{len(engine.bones)} 插槽:{len(engine.skel['slots'])} "
                              f"动画:{len(engine.animations)} 版本:{SPINE_VERSION}")

    def _start_batch(self):
        # If batch list is visible, use checked items from the list
        if self.batch_file_list.isVisible():
            checked = self._get_checked_paths()
            if not checked:
                QMessageBox.warning(self, "提示", "请先在列表中勾选要导出的文件，或使用「从文件夹智能导入」添加文件。")
                return
        elif not self.batch_sources:
            QMessageBox.warning(self, "提示", "请先添加批量文件")
            return
        else:
            checked = list(self.batch_sources)

        out_base = self.folder_output_dir if self.folder_output_dir else self.output_dir
        if out_base is None:
            # Default: parent of first file
            parent = checked[0].parent
            out_base = parent / "spine_batch"
            self._log(f"默认输出目录: {out_base}")

        out_base.mkdir(parents=True, exist_ok=True)

        editor = self.cmb_mode.currentIndex() == 0

        # Set TARGET_W
        import unity_to_spine
        unity_to_spine.TARGET_W = self.spin_target_w.value()

        self._disable_all_buttons()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(checked))
        self.progress_bar.setValue(0)
        self._log(f"开始批量导出 {len(checked)} 个文件...")

        self.batch_worker = BatchConvertWorker(checked, out_base, editor)
        self.batch_worker.progress.connect(self._log)
        self.batch_worker.file_progress.connect(self._on_batch_progress)
        self.batch_worker.finished_signal.connect(self._on_batch_done)
        self.batch_worker.start()
    
    def _on_batch_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)

    def _on_batch_done(self, success: int, fail: int):
        self._enable_all_buttons()
        self.progress_bar.setVisible(False)
        self._log(f"✅ 批量导出完成: {success} 成功, {fail} 失败")
        self.lbl_info.setText(f"批量完成: {success}/{success + fail}")
        QMessageBox.information(
            self, "批量导出完成",
            f"成功: {success}\n失败: {fail}\n总计: {success + fail}"
        )

    # ---- Preview controls ----

    def _toggle_play(self):
        if not self.preview_engine:
            return
        if self.canvas.playing:
            self.canvas.pause()
            self.btn_play.setText("▶")
        else:
            self.canvas.play()
            self.btn_play.setText("⏸")

    def _stop_anim(self):
        if not self.preview_engine:
            return
        self.canvas.stop()
        self.btn_play.setText("▶")
        self.sld_time.setValue(0)
        self.lbl_time.setText("0.00s")

    def _toggle_loop(self, checked: bool):
        self.canvas.loop = checked

    def _speed_changed(self, val: float):
        self.canvas.speed = val

    def _anim_changed(self, name: str):
        if not self.preview_engine or not name:
            return
        self.preview_engine.select_animation(name)
        self.canvas.time = 0.0
        self.canvas.playing = False
        self.btn_play.setText("▶")
        self.sld_time.setValue(0)
        self.lbl_time.setText("0.00s")
        self.canvas._request_render(0.0, target_w=self.canvas._needed_render_width())
        self.canvas.update()

    def _slider_pressed(self):
        self._was_playing = self.canvas.playing
        self.canvas.pause()

    def _slider_released(self):
        if getattr(self, "_was_playing", False):
            self.canvas.play()
            self.btn_play.setText("⏸")

    def _slider_moved(self, val: int):
        if not self.preview_engine or not self.preview_engine.current_anim:
            return
        duration = self.preview_engine.anim_stop
        if duration <= 0:
            duration = 1.0
        t = val / 1000.0 * duration
        self.canvas.time = t
        self.canvas._last_tick = time.time()
        self.lbl_time.setText(f"{t:.2f}s")
        self.canvas._request_render(t, target_w=self.canvas._needed_render_width())
        self.canvas.update()

    def _toggle_skeleton(self, checked: bool):
        if self.preview_engine:
            self.preview_engine.show_skeleton = checked
        self.canvas.update()

    # ---- Logging ----

    def _log(self, msg: str):
        self.txt_log.append(msg)
        # Auto-scroll
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())


# =============  Entry Point  =============

def main():
    # Set OpenGL surface format before creating QApplication
    if HAS_OPENGL:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSamples(0)
        QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    # Dark palette fallback
    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(13, 13, 26))
    palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 50))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(30, 42, 78))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.Highlight, QColor(233, 69, 96))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
