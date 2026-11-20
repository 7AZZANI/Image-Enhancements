"""Zero-dependency local browser UI (HTTP server + drag-and-drop split-slider SPA + watch control)."""

from __future__ import annotations

import base64
from dataclasses import replace
import hmac
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np

from . import __version__
from .automation import WatchFolderRunner
from .benchmark import run_doctor
from .image_io import INPUT_EXTENSIONS, SUPPORTED_EXTENSIONS, load_image, save_image
from .inference import EnginePool
from .models import MODEL_REGISTRY
from .profiles import apply_profile_to_config, list_profiles
from .server import AsyncJobManager, dispatch_webhook
from .pipeline import (
    PRESET_MODEL_MAP,
    EnhancementConfig,
    EnhancementResult,
    PreparedEnhancementLayers,
    _resolve_tone_config,
    compose_enhancement_result,
    prepare_enhancement_layers,
)

MAX_BODY_BYTES = 80 * 1024 * 1024
SAFE_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AZZANI Image Enhancements — GPU Studio</title>
<style>
  :root {
    --bg: #080c14;
    --panel: #101726;
    --panel-2: #162035;
    --border: #253452;
    --text: #eef4ff;
    --muted: #93a4c7;
    --accent: #38bdf8;
    --accent-2: #22d3ee;
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #fb7185;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at top right, #13213c 0%, var(--bg) 55%);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    min-height: 100vh;
  }
  .wrap { max-width: 1260px; margin: 0 auto; padding: 22px 18px 48px; }
  header {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 14px; margin-bottom: 18px;
  }
  .brand h1 { margin: 0; font-size: 1.45rem; letter-spacing: 0.02em; }
  .brand span { color: var(--accent); }
  .brand p { margin: 4px 0 0; color: var(--muted); font-size: 0.9rem; }
  .gpu-badge {
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 0.85rem;
    line-height: 1.4;
  }
  .gpu-badge strong { color: var(--accent-2); }
  .grid { display: grid; grid-template-columns: 350px 1fr; gap: 18px; }
  @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: rgba(16, 23, 38, 0.92);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
  }
  .card h2 { margin: 0 0 12px; font-size: 1.02rem; color: #dbeafe; }
  label { display: block; font-size: 0.82rem; color: var(--muted); margin: 10px 0 4px; }
  select, input[type="text"], input[type="number"], input[type="range"] {
    width: 100%;
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 8px 10px;
    font-size: 0.9rem;
  }
  input[type="range"] { padding: 4px 0; border: none; background: transparent; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .btn {
    width: 100%;
    margin-top: 14px;
    background: linear-gradient(135deg, var(--accent), #0284c7);
    color: #04121f;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 0.95rem;
    cursor: pointer;
    transition: transform 0.1s ease, opacity 0.15s ease;
  }
  .btn:hover:not(:disabled) { transform: translateY(-1px); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-sec {
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    margin-top: 8px;
  }
  .btn-danger {
    background: rgba(251, 113, 133, 0.16);
    color: #fecdd3;
    border: 1px solid rgba(251, 113, 133, 0.45);
  }
  .drop {
    border: 2px dashed #334b77;
    border-radius: 14px;
    padding: 26px 16px;
    text-align: center;
    cursor: pointer;
    background: rgba(22, 32, 53, 0.55);
    transition: border-color 0.15s, background 0.15s;
  }
  .drop.over { border-color: var(--accent); background: rgba(56, 189, 248, 0.10); }
  .drop p { margin: 6px 0 0; color: var(--muted); font-size: 0.85rem; }
  .compare-box {
    position: relative;
    width: 100%;
    min-height: 400px;
    max-height: 70vh;
    overflow: hidden;
    border-radius: 14px;
    background: repeating-conic-gradient(#0d1422 0% 25%, #121c2e 0% 50%) 50% / 20px 20px;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
  }
  .compare-stage { position: relative; display: inline-block; max-width: 100%; max-height: 68vh; }
  .compare-stage img { display: block; max-width: 100%; max-height: 68vh; width: auto; height: auto; }
  .after-layer {
    position: absolute; inset: 0; overflow: hidden;
  }
  .after-layer img { width: 100%; height: 100%; object-fit: contain; }
  .divider {
    position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.85); pointer-events: none;
  }
  .handle {
    position: absolute; top: 50%; transform: translate(-50%, -50%);
    width: 34px; height: 34px; border-radius: 50%;
    background: var(--accent); color: #04121f; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45); pointer-events: none; font-size: 0.8rem;
  }
  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 10px; margin-top: 14px;
  }
  .stat {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px;
  }
  .stat .k { font-size: 0.74rem; color: var(--muted); text-transform: uppercase; }
  .stat .v { font-size: 0.96rem; font-weight: 600; margin-top: 3px; }
  .status-line { margin-top: 10px; font-size: 0.86rem; color: var(--accent-2); min-height: 20px; }
  .pill { display: inline-block; font-size: 0.75rem; padding: 2px 8px; border-radius: 999px; background: rgba(52, 211, 153, 0.16); color: var(--good); border: 1px solid rgba(52, 211, 153, 0.35); }
  .watch-log { margin-top: 10px; max-height: 150px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 0.78rem; color: var(--muted); background: #0a101b; border-radius: 8px; padding: 8px; border: 1px solid var(--border); }
  .slider-head { display: flex; justify-content: space-between; align-items: baseline; margin: 8px 0 2px; font-size: 0.8rem; color: var(--muted); }
  .slider-head .val { font-family: ui-monospace, monospace; color: var(--accent-2); font-size: 0.78rem; }
  .check-row { display: flex; align-items: center; gap: 8px; margin: 10px 0 4px; font-size: 0.82rem; color: var(--muted); cursor: pointer; }
  .check-row input { width: auto; accent-color: var(--accent); }
  .loupe-box {
    display: none; position: absolute; z-index: 25; pointer-events: none;
    background: rgba(8, 12, 20, 0.96); border: 1px solid var(--accent);
    border-radius: 12px; padding: 6px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
  }
  .loupe-pair { display: grid; grid-template-columns: 140px 140px; gap: 6px; }
  .loupe-pane { text-align: center; }
  .loupe-pane canvas { display: block; width: 140px; height: 140px; border-radius: 6px; border: 1px solid var(--border); image-rendering: pixelated; background: #090e19; }
  .loupe-pane span { display: block; font-size: 0.68rem; color: var(--muted); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.04em; }
  .hist-wrap {
    margin-top: 12px; background: #090e19; border: 1px solid var(--border);
    border-radius: 12px; padding: 10px 12px;
  }
  .hist-head { display: flex; justify-content: space-between; align-items: center; font-size: 0.76rem; color: var(--muted); margin-bottom: 6px; }
  .hist-wrap canvas { width: 100%; height: 108px; display: block; }
  .btn-toggle.active { border-color: var(--accent); color: var(--accent-2); background: rgba(56, 189, 248, 0.14); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <h1><span>AZZANI</span> Image Enhancements</h1>
      <p>Open-Source (MIT) Hybrid AI + 32-bit Photometric Studio &bull; Local GPU Accelerated</p>
    </div>
    <div id="gpuBadge" class="gpu-badge">Detecting local GPU...</div>
  </header>

  <div class="grid">
    <div>
      <div class="card">
        <h2>1. Upload &amp; Settings</h2>
        <div id="dropZone" class="drop">
          <strong>Drop image(s) here or click to browse</strong>
          <p>PNG, JPEG, WebP, TIFF, BMP, Camera RAW (.dng/.cr2/.nef/.arw) &bull; EXIF &amp; Alpha safe</p>
          <input id="fileInput" type="file" accept="image/*,.dng,.cr2,.cr3,.nef,.arw,.raf,.orf,.rw2,.pef" multiple style="display:none">
        </div>
        <div id="queueText" class="status-line"></div>

        <label for="preset">Enhancement Preset</label>
        <select id="preset">
          <option value="auto" selected>Auto (Adaptive GPU + Image)</option>
          <option value="fast">Fast (2x SPAN)</option>
          <option value="pro">Pro (4x RealPLKSR)</option>
          <option value="ultra">Ultra (4x DAT2 Transformer)</option>
          <option value="hq">HQ (4x Clean Studio DAT2)</option>
          <option value="portrait">Portrait (4x FaceDAT + GFPGAN)</option>
          <option value="anime">Anime / Illustration (4x)</option>
          <option value="anime-2x">Anime / Illustration (2x)</option>
          <option value="deblur">Deblur 1x (NAFNet Motion Rescue)</option>
          <option value="restore">Restore 1x (De-JPEG + Photometric)</option>
          <option value="photometric">Photometric Only (Zero-AI DSP)</option>
        </select>

        <div class="row2">
          <div>
            <label for="scale">Scale</label>
            <select id="scale">
              <option value="">Auto</option>
              <option value="1">1x</option>
              <option value="2">2x</option>
              <option value="4">4x</option>
            </select>
          </div>
          <div>
            <label for="tta">Self-Ensemble (TTA)</label>
            <select id="tta">
              <option value="1" selected>1x (Fast)</option>
              <option value="2">2x</option>
              <option value="4">4x (Ultra Crisp)</option>
              <option value="8">8x (Max)</option>
            </select>
          </div>
        </div>

        <div class="row2">
          <div>
            <label for="faceRestore">Face Restoration</label>
            <select id="faceRestore">
              <option value="auto" selected>Auto (YuNet + GFPGAN)</option>
              <option value="on">Force On</option>
              <option value="off">Off</option>
            </select>
          </div>
          <div>
            <label for="outExt">Output Format</label>
            <select id="outExt">
              <option value="png" selected>PNG</option>
              <option value="jpg">JPEG (4:4:4 Q98)</option>
              <option value="webp">WebP</option>
              <option value="tiff">TIFF</option>
            </select>
          </div>
        </div>

        <div class="row2">
          <div>
            <label for="bitDepth">Bit Depth</label>
            <select id="bitDepth">
              <option value="8" selected>8-bit (TPDF Dithered)</option>
              <option value="16">16-bit (PNG/TIFF Master)</option>
            </select>
          </div>
          <div>
            <label for="backend">Execution Backend</label>
            <select id="backend">
              <option value="auto" selected>Auto (Best GPU / DirectML)</option>
              <option value="pytorch">PyTorch Eager</option>
              <option value="compile">torch.compile (CUDA Inductor)</option>
              <option value="onnx">ONNX Runtime</option>
              <option value="directml">DirectML (Windows AMD/Intel)</option>
              <option value="tensorrt">TensorRT EP</option>
            </select>
          </div>
        </div>

        <div class="slider-head"><label for="creativity" style="margin:0;">Hybrid Creativity (Gen Detail)</label><span id="valCreativity" class="val">0.00</span></div>
        <input id="creativity" type="range" min="0" max="1" step="0.05" value="0">

        <button id="enhanceBtn" class="btn" disabled>Enhance Selected Image(s)</button>
        <a id="downloadLink" class="btn btn-sec" style="display:none; text-align:center; text-decoration:none;" download>Download Enhanced Output</a>
      </div>

      <div id="retuneCard" class="card" style="margin-top: 16px; display:none;">
        <h2 style="display:flex; justify-content:space-between; align-items:center;">
          <span>2. Live Studio Retuning</span>
          <span id="retuneTiming" class="pill">instant</span>
        </h2>

        <div class="slider-head"><span>Tone / HDR Recovery</span><span id="valTone" class="val">0.20</span></div>
        <input id="rtTone" type="range" min="0" max="1" step="0.02" value="0.20">

        <div class="slider-head"><span>Guided Detail Sharpness</span><span id="valDetail" class="val">0.12</span></div>
        <input id="rtDetail" type="range" min="0" max="1.2" step="0.02" value="0.12">

        <div class="slider-head"><span>Wavelet Color Fidelity</span><span id="valColor" class="val">0.85</span></div>
        <input id="rtColor" type="range" min="0" max="1" step="0.02" value="0.85">

        <div class="slider-head"><span>White Balance Strength</span><span id="valWb" class="val">0.65</span></div>
        <input id="rtWb" type="range" min="0" max="1" step="0.02" value="0.65">

        <div class="slider-head"><span>Chromatic Aberration Removal</span><span id="valCa" class="val">0.45</span></div>
        <input id="rtCa" type="range" min="0" max="1" step="0.02" value="0.45">

        <div class="slider-head"><span>Self-Critic Halo Guard</span><span id="valHalo" class="val">0.75</span></div>
        <input id="rtHalo" type="range" min="0" max="1" step="0.02" value="0.75">

        <div class="slider-head"><span>Face Fidelity (GFPGAN)</span><span id="valFace" class="val">0.78</span></div>
        <input id="rtFace" type="range" min="0" max="1" step="0.02" value="0.78">

        <div class="slider-head"><span>Natural Micro-Grain</span><span id="valGrain" class="val">0.000</span></div>
        <input id="rtGrain" type="range" min="0" max="0.015" step="0.001" value="0.000">

        <label class="check-row"><input id="rtBokeh" type="checkbox" checked> Bokeh / DoF Focus Guard (shield blurry background)</label>
        <button id="rtResetBtn" type="button" class="btn btn-sec">Reset to AI Auto</button>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h2>3. Hot-Folder Automation <span id="watchPill" class="pill" style="display:none;">RUNNING</span></h2>
        <label for="watchIn">Watch Inbox Folder</label>
        <input id="watchIn" type="text" value="./azzani_inbox">
        <label for="watchOut">Enhanced Output Folder</label>
        <input id="watchOut" type="text" value="./azzani_enhanced">
        <div class="row2">
          <button id="watchStartBtn" class="btn">Start Automation</button>
          <button id="watchStopBtn" class="btn btn-danger" disabled>Stop</button>
        </div>
        <div id="watchLog" class="watch-log">Drop photos into the inbox folder to enhance them automatically using warm GPU weights.</div>
      </div>
    </div>

    <div class="card">
      <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px;">
        <h2 style="margin:0;">Before / After Split Preview</h2>
        <div style="display:flex; align-items:center; gap:10px;">
          <button id="loupeBtn" type="button" class="btn btn-sec btn-toggle" style="width:auto; margin:0; padding:6px 11px; font-size:0.78rem;">4x Pixel Loupe: OFF</button>
          <div style="width:185px;">
            <input id="splitRange" type="range" min="0" max="100" value="50" title="Drag to compare Before vs After">
          </div>
        </div>
      </div>

      <div id="compareBox" class="compare-box">
        <div id="emptyHint" style="color:var(--muted); text-align:center; padding:24px;">
          Upload an image on the left to inspect Before / After restoration quality on your GPU.
        </div>
        <div id="stage" class="compare-stage" style="display:none;">
          <img id="beforeImg" alt="Before image">
          <div id="afterLayer" class="after-layer" style="clip-path: inset(0 0 0 50%);">
            <img id="afterImg" alt="Enhanced image">
          </div>
          <div id="divider" class="divider" style="left:50%;"></div>
          <div id="handle" class="handle" style="left:50%;">&#8596;</div>
        </div>
        <div id="loupeBox" class="loupe-box">
          <div class="loupe-pair">
            <div class="loupe-pane"><canvas id="loupeBefore" width="140" height="140"></canvas><span>Before (4x)</span></div>
            <div class="loupe-pane"><canvas id="loupeAfter" width="140" height="140"></canvas><span>After (4x)</span></div>
          </div>
        </div>
      </div>

      <div id="histPanel" class="hist-wrap" style="display:none;">
        <div class="hist-head">
          <span>Output RGB &amp; Luminance Distribution (32-bit Photometric Check)</span>
          <span id="histMeta" style="font-family: ui-monospace, monospace;">64 bins</span>
        </div>
        <canvas id="histCanvas" width="640" height="108"></canvas>
      </div>

      <div id="stats" class="stats" style="display:none;"></div>
    </div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let selectedFiles = [];
let activeJobId = null;
let defaultRetuneValues = null;
let loupeEnabled = false;
let retuneTimer = null;
let retuneInFlight = false;
let retuneQueued = false;

const SLIDER_SPECS = [
  ["rtTone", "valTone", "tone", 2],
  ["rtDetail", "valDetail", "detail", 2],
  ["rtColor", "valColor", "color_fix", 2],
  ["rtWb", "valWb", "wb_strength", 2],
  ["rtCa", "valCa", "ca_suppress", 2],
  ["rtHalo", "valHalo", "halo_guard", 2],
  ["rtFace", "valFace", "face_fidelity", 2],
  ["rtGrain", "valGrain", "grain", 3],
];

function syncSliderLabels() {
  for (const [inputId, valId, , dec] of SLIDER_SPECS) {
    $(valId).textContent = Number($(inputId).value).toFixed(dec);
  }
}

function applyRetuneDefaults(defs) {
  if (!defs) return;
  for (const [inputId, , key] of SLIDER_SPECS) {
    if (defs[key] !== undefined && defs[key] !== null) {
      $(inputId).value = String(defs[key]);
    }
  }
  if (defs.bokeh_guard !== undefined) {
    $("rtBokeh").checked = !!defs.bokeh_guard;
  }
  syncSliderLabels();
}

function updateSplit(pct) {
  const clamped = Math.max(0, Math.min(100, Number(pct)));
  $("afterLayer").style.clipPath = `inset(0 0 0 ${clamped}%)`;
  $("divider").style.left = `${clamped}%`;
  $("handle").style.left = `${clamped}%`;
}

function hideLoupe() {
  $("loupeBox").style.display = "none";
}

function drawLoupeAt(clientX, clientY) {
  if (!loupeEnabled) { hideLoupe(); return; }
  const bImg = $("beforeImg");
  const aImg = $("afterImg");
  if (!bImg.naturalWidth || !aImg.naturalWidth) { hideLoupe(); return; }

  const stageRect = $("stage").getBoundingClientRect();
  const boxRect = $("compareBox").getBoundingClientRect();
  if (stageRect.width <= 0 || stageRect.height <= 0) return;

  const nx = Math.max(0, Math.min(1, (clientX - stageRect.left) / stageRect.width));
  const ny = Math.max(0, Math.min(1, (clientY - stageRect.top) / stageRect.height));

  const lensSize = 140;
  const aCropW = Math.min(aImg.naturalWidth, Math.max(8, lensSize / 4));
  const aCropH = Math.min(aImg.naturalHeight, Math.max(8, lensSize / 4));
  const bCropW = Math.min(bImg.naturalWidth, Math.max(4, (aCropW / aImg.naturalWidth) * bImg.naturalWidth));
  const bCropH = Math.min(bImg.naturalHeight, Math.max(4, (aCropH / aImg.naturalHeight) * bImg.naturalHeight));

  const ax = Math.max(0, Math.min(aImg.naturalWidth - aCropW, nx * aImg.naturalWidth - aCropW / 2));
  const ay = Math.max(0, Math.min(aImg.naturalHeight - aCropH, ny * aImg.naturalHeight - aCropH / 2));
  const bx = Math.max(0, Math.min(bImg.naturalWidth - bCropW, nx * bImg.naturalWidth - bCropW / 2));
  const by = Math.max(0, Math.min(bImg.naturalHeight - bCropH, ny * bImg.naturalHeight - bCropH / 2));

  const bCtx = $("loupeBefore").getContext("2d");
  const aCtx = $("loupeAfter").getContext("2d");
  bCtx.clearRect(0, 0, lensSize, lensSize);
  aCtx.clearRect(0, 0, lensSize, lensSize);
  bCtx.imageSmoothingEnabled = false;
  aCtx.imageSmoothingEnabled = false;
  bCtx.drawImage(bImg, bx, by, bCropW, bCropH, 0, 0, lensSize, lensSize);
  aCtx.drawImage(aImg, ax, ay, aCropW, aCropH, 0, 0, lensSize, lensSize);

  const loupe = $("loupeBox");
  loupe.style.display = "block";
  const lw = loupe.offsetWidth || 300;
  const lh = loupe.offsetHeight || 175;
  let px = clientX - boxRect.left + 16;
  let py = clientY - boxRect.top + 16;
  if (px + lw > boxRect.width - 8) px = clientX - boxRect.left - lw - 16;
  if (py + lh > boxRect.height - 8) py = clientY - boxRect.top - lh - 16;
  loupe.style.left = `${Math.max(6, px)}px`;
  loupe.style.top = `${Math.max(6, py)}px`;
}

function drawHistogram(hist) {
  if (!hist || !hist.bins) {
    $("histPanel").style.display = "none";
    return;
  }
  $("histPanel").style.display = "block";
  const canvas = $("histCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(147, 164, 199, 0.14)";
  ctx.lineWidth = 1;
  for (let g = 1; g <= 3; g++) {
    const gx = Math.round((w * g) / 4) + 0.5;
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
  }

  const drawCurve = (vals, stroke, fill) => {
    if (!vals || !vals.length) return;
    const n = vals.length;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const y = h - Math.max(0, Math.min(1, vals[i])) * (h - 6);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const y = h - Math.max(0, Math.min(1, vals[i])) * (h - 6);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  };

  drawCurve(hist.r, "rgba(251, 113, 133, 0.85)", "rgba(251, 113, 133, 0.18)");
  drawCurve(hist.g, "rgba(52, 211, 153, 0.85)", "rgba(52, 211, 153, 0.16)");
  drawCurve(hist.b, "rgba(56, 189, 248, 0.85)", "rgba(56, 189, 248, 0.18)");
  drawCurve(hist.luma, "rgba(238, 244, 255, 0.92)", null);
}

$("splitRange").addEventListener("input", (e) => updateSplit(e.target.value));
$("stage").addEventListener("pointermove", (e) => {
  if (e.buttons === 1 || e.pointerType !== "mouse") {
    const rect = $("stage").getBoundingClientRect();
    if (rect.width > 0) {
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      $("splitRange").value = Math.round(pct);
      updateSplit(pct);
    }
  }
  drawLoupeAt(e.clientX, e.clientY);
});
$("stage").addEventListener("pointerleave", hideLoupe);

$("loupeBtn").addEventListener("click", () => {
  loupeEnabled = !loupeEnabled;
  $("loupeBtn").textContent = `4x Pixel Loupe: ${loupeEnabled ? "ON" : "OFF"}`;
  $("loupeBtn").classList.toggle("active", loupeEnabled);
  if (!loupeEnabled) hideLoupe();
});

async function refreshDoctor() {
  try {
    const res = await fetch("/api/doctor");
    const d = await res.json();
    const badge = $("gpuBadge");
    badge.textContent = "";
    const title = document.createElement("strong");
    if (d.gpu_name) {
      title.textContent = `${d.gpu_name}`;
      badge.appendChild(title);
      const vram = d.vram_total_mb ? ` (${(d.vram_free_mb/1024).toFixed(1)}/${(d.vram_total_mb/1024).toFixed(1)} GB VRAM free)` : "";
      badge.appendChild(document.createTextNode(` • ${d.active_device || "cuda"}${vram}`));
    } else {
      title.textContent = `CPU Mode (${d.platform})`;
      badge.appendChild(title);
    }
    const sub = document.createElement("div");
    sub.style.color = "var(--muted)";
    sub.textContent = d.recommendation || d.integration_status || "";
    badge.appendChild(sub);
  } catch (err) {
    $("gpuBadge").textContent = "Hardware check unavailable";
  }
}

function setFiles(fileList) {
  selectedFiles = Array.from(fileList || []).filter((f) => f.size > 0);
  $("enhanceBtn").disabled = selectedFiles.length === 0;
  $("queueText").textContent = selectedFiles.length
    ? `Ready: ${selectedFiles.map((f) => f.name).join(", ")}`
    : "";
}

$("dropZone").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => setFiles(e.target.files));
$("dropZone").addEventListener("dragover", (e) => { e.preventDefault(); $("dropZone").classList.add("over"); });
$("dropZone").addEventListener("dragleave", () => $("dropZone").classList.remove("over"));
$("dropZone").addEventListener("drop", (e) => {
  e.preventDefault();
  $("dropZone").classList.remove("over");
  setFiles(e.dataTransfer.files);
});

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Read failed"));
    reader.readAsDataURL(file);
  });
}

function renderStats(summary) {
  const s = $("stats");
  s.textContent = "";
  const cm = summary.critic_metrics || {};
  const conf = Number(summary.quality_confidence ?? 100);
  const halo = Number(cm.halo_ringing_index ?? 0);
  const de = Number(cm.color_drift_delta_e ?? 0);
  const idCos = cm.face_identity_cosine !== undefined ? ` • ID cos=${Number(cm.face_identity_cosine).toFixed(3)}` : "";
  const cards = [
    ["Quality Confidence", `${conf.toFixed(1)}%`],
    ["Self-Critic Guard", `halo=${halo.toFixed(3)} • ΔE=${de.toFixed(1)}${idCos}`],
    ["Resolution", `${summary.input_size[0]}x${summary.input_size[1]} -> ${summary.output_size[0]}x${summary.output_size[1]}`],
    ["Sharpness", `${summary.before.sharpness_score.toFixed(2)} -> ${summary.after.sharpness_score.toFixed(2)}`],
    ["JPEG Blockiness", `${summary.before.jpeg_blockiness.toFixed(2)} -> ${summary.after.jpeg_blockiness.toFixed(2)}`],
    ["Models", (summary.models_used || []).join(", ") || "photometric"],
    ["GPU / Precision", `${summary.device_used} • ${summary.precision_used}`],
    ["Time / Tile", `${summary.elapsed_seconds.toFixed(2)}s (tile=${summary.tile_size_used || "full"})`],
  ];
  for (const [k, v] of cards) {
    const box = document.createElement("div");
    box.className = "stat";
    const kd = document.createElement("div"); kd.className = "k"; kd.textContent = k;
    const vd = document.createElement("div"); vd.className = "v"; vd.textContent = v;
    box.appendChild(kd); box.appendChild(vd);
    s.appendChild(box);
  }
  s.style.display = "grid";
}

function applyEnhanceResponse(data) {
  activeJobId = data.job_id;
  $("emptyHint").style.display = "none";
  $("stage").style.display = "inline-block";
  $("beforeImg").src = data.before_url;
  $("afterImg").src = data.preview_url || data.result_url;
  updateSplit($("splitRange").value);
  const sep = (data.result_url || "").includes("?") ? "&" : "?";
  $("downloadLink").href = `${data.result_url}${sep}download=1`;
  $("downloadLink").download = data.download_name;
  $("downloadLink").style.display = "block";
  $("retuneCard").style.display = "block";
  if (data.retune_defaults) {
    defaultRetuneValues = data.retune_defaults;
    applyRetuneDefaults(data.retune_defaults);
  }
  renderStats(data.summary);
  drawHistogram(data.histogram);
}

async function executeRetune() {
  if (!activeJobId) return;
  if (retuneInFlight) { retuneQueued = true; return; }
  retuneInFlight = true;
  try {
    const payload = {
      job_id: activeJobId,
      tone: Number($("rtTone").value),
      detail: Number($("rtDetail").value),
      color_fix: Number($("rtColor").value),
      wb_strength: Number($("rtWb").value),
      ca_suppress: Number($("rtCa").value),
      halo_guard: Number($("rtHalo").value),
      face_fidelity: Number($("rtFace").value),
      grain: Number($("rtGrain").value),
      bokeh_guard: $("rtBokeh").checked,
      output_ext: $("outExt").value,
      bit_depth: Number($("bitDepth").value),
    };
    const resp = await fetch("/api/retune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    $("afterImg").src = data.preview_url || data.result_url;
    const sep = (data.result_url || "").includes("?") ? "&" : "?";
    $("downloadLink").href = `${data.result_url}${sep}download=1`;
    $("downloadLink").download = data.download_name;
    renderStats(data.summary);
    drawHistogram(data.histogram);
    $("retuneTiming").textContent = `${Math.max(1, Math.round(data.summary.elapsed_seconds * 1000))} ms`;
  } catch (err) {
    $("queueText").textContent = `Retune warning: ${err.message}`;
  } finally {
    retuneInFlight = false;
    if (retuneQueued) {
      retuneQueued = false;
      executeRetune();
    }
  }
}

function scheduleRetune(delayMs = 55) {
  syncSliderLabels();
  if (!activeJobId) return;
  clearTimeout(retuneTimer);
  retuneTimer = setTimeout(executeRetune, delayMs);
}

for (const [inputId] of SLIDER_SPECS) {
  $(inputId).addEventListener("input", () => scheduleRetune(55));
}
$("rtBokeh").addEventListener("change", () => scheduleRetune(0));
$("outExt").addEventListener("change", () => scheduleRetune(0));
$("bitDepth").addEventListener("change", () => scheduleRetune(0));
$("rtResetBtn").addEventListener("click", () => {
  applyRetuneDefaults(defaultRetuneValues);
  scheduleRetune(0);
});

$("creativity").addEventListener("input", () => {
  $("valCreativity").textContent = Number($("creativity").value).toFixed(2);
});

$("enhanceBtn").addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  $("enhanceBtn").disabled = true;
  $("downloadLink").style.display = "none";

  try {
    for (let i = 0; i < selectedFiles.length; i++) {
      const f = selectedFiles[i];
      $("queueText").textContent = `Processing ${i + 1}/${selectedFiles.length}: ${f.name} on GPU...`;
      const b64 = await readFileAsBase64(f);
      const scaleVal = $("scale").value ? Number($("scale").value) : null;
      const payload = {
        filename: f.name,
        data_base64: b64,
        preset: $("preset").value,
        scale: scaleVal,
        tta: Number($("tta").value),
        face_restore: $("faceRestore").value,
        output_ext: $("outExt").value,
        bit_depth: Number($("bitDepth").value),
        backend: $("backend").value,
        creativity: Number($("creativity").value),
      };
      const resp = await fetch("/api/enhance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      applyEnhanceResponse(data);
      $("queueText").textContent = `Done (${i + 1}/${selectedFiles.length}): ${data.download_name} (${data.summary.elapsed_seconds.toFixed(2)}s)`;
    }
    refreshDoctor();
  } catch (err) {
    $("queueText").textContent = `Error: ${err.message}`;
  } finally {
    $("enhanceBtn").disabled = false;
  }
});

async function refreshWatchStatus() {
  try {
    const res = await fetch("/api/watch/status");
    const st = await res.json();
    const active = !!st.running;
    $("watchPill").style.display = active ? "inline-block" : "none";
    $("watchStartBtn").disabled = active;
    $("watchStopBtn").disabled = !active;
    if (st.watch_dir) {
      const lines = [`Status: ${active ? "WATCHING" : "STOPPED"} | Processed: ${st.processed_count || 0} | Errors: ${st.error_count || 0}`];
      for (const ev of (st.recent_events || [])) {
        lines.push(`[${ev.status}] ${ev.input_path} -> ${ev.output_path || ev.error || ""}`);
      }
      $("watchLog").textContent = lines.join("\n");
    }
  } catch (e) {}
}

$("watchStartBtn").addEventListener("click", async () => {
  const resp = await fetch("/api/watch/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      watch_dir: $("watchIn").value,
      output_dir: $("watchOut").value,
      preset: $("preset").value,
      tta: Number($("tta").value),
    }),
  });
  await resp.json();
  refreshWatchStatus();
});

$("watchStopBtn").addEventListener("click", async () => {
  await fetch("/api/watch/stop", { method: "POST" });
  refreshWatchStatus();
});

refreshDoctor();
refreshWatchStatus();
setInterval(refreshWatchStatus, 2500);
</script>
</body>
</html>
"""


def _compute_histogram(rgb_f32: np.ndarray, bins: int = 64) -> Dict[str, Any]:
    """Compute compact 64-bin normalized R, G, B, and BT.709 luminance distributions."""
    h, w = rgb_f32.shape[:2]
    step = max(1, int(math.sqrt(max(1, h * w) / 250_000.0)))
    sample = rgb_f32[::step, ::step]
    u8 = np.clip(np.rint(sample * 255.0), 0, 255).astype(np.uint8)
    luma_f = 0.2126 * sample[:, :, 0] + 0.7152 * sample[:, :, 1] + 0.0722 * sample[:, :, 2]
    luma_u8 = np.clip(np.rint(luma_f * 255.0), 0, 255).astype(np.uint8)
    bin_width = 256 // bins

    r = np.bincount(u8[:, :, 0].ravel(), minlength=256).reshape(bins, bin_width).sum(axis=1).astype(np.float32)
    g = np.bincount(u8[:, :, 1].ravel(), minlength=256).reshape(bins, bin_width).sum(axis=1).astype(np.float32)
    b = np.bincount(u8[:, :, 2].ravel(), minlength=256).reshape(bins, bin_width).sum(axis=1).astype(np.float32)
    luma = np.bincount(luma_u8.ravel(), minlength=256).reshape(bins, bin_width).sum(axis=1).astype(np.float32)

    peak = max(float(r.max()), float(g.max()), float(b.max()), float(luma.max()), 1.0)
    return {
        "bins": bins,
        "r": [round(float(x / peak), 4) for x in r],
        "g": [round(float(x / peak), 4) for x in g],
        "b": [round(float(x / peak), 4) for x in b],
        "luma": [round(float(x / peak), 4) for x in luma],
    }


def _build_retune_defaults(
    layers: PreparedEnhancementLayers,
    cfg: EnhancementConfig,
    out_ext: str,
) -> Dict[str, Any]:
    resolved = _resolve_tone_config(layers.diag_before, cfg, layers.primary_model)
    return {
        "tone": round(float(resolved.tone_strength), 2),
        "detail": round(float(resolved.detail_strength), 2),
        "color_fix": round(float(cfg.color_fix if layers.primary_model else 0.0), 2),
        "wb_strength": round(float(resolved.white_balance_strength), 2),
        "ca_suppress": round(float(resolved.ca_suppress), 2),
        "halo_guard": round(float(resolved.halo_guard), 2),
        "face_fidelity": round(float(cfg.face_fidelity), 2),
        "grain": round(float(resolved.natural_grain), 3),
        "bokeh_guard": bool(resolved.bokeh_guard),
        "output_ext": out_ext.lstrip("."),
        "bit_depth": int(cfg.bit_depth),
    }


def _write_job_outputs(
    out_path: Path,
    preview_png: Path,
    out_ext: str,
    res: EnhancementResult,
    cfg: EnhancementConfig,
    exif_bytes: Optional[bytes],
    icc_profile: Optional[bytes],
) -> Path:
    save_image(
        out_path,
        res.output_rgb,
        alpha=res.output_alpha,
        bit_depth=cfg.bit_depth,
        jpeg_quality=cfg.jpeg_quality,
        webp_lossless=cfg.webp_lossless,
        dither=cfg.dither,
        exif_bytes=exif_bytes if cfg.preserve_metadata else None,
        icc_profile=icc_profile if cfg.preserve_metadata else None,
    )
    if out_ext in {".tif", ".tiff"} or cfg.bit_depth == 16:
        save_image(preview_png, res.output_rgb, alpha=res.output_alpha, bit_depth=8, dither=False)
        return preview_png
    return out_path


class WebUIState:
    """Thread-safe backend state shared across HTTP handlers."""

    def __init__(
        self,
        *,
        models_dir: Optional[Path] = None,
        max_jobs: int = 8,
        api_key: Optional[str] = None,
    ) -> None:
        self.models_dir = models_dir
        self.max_jobs = max_jobs
        self.api_key = api_key if api_key is not None else (os.environ.get("AZZANI_API_KEY") or None)
        self.engine_pool = EnginePool(max_models=2)
        self.infer_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.temp_dir = tempfile.TemporaryDirectory(prefix="azzani_webui_")
        self.temp_root = Path(self.temp_dir.name)
        self.watcher: Optional[WatchFolderRunner] = None
        self.watch_lock = threading.Lock()
        self.async_jobs = AsyncJobManager(
            lambda spec: _process_enhance_request(
                self,
                spec,
                fixed_job_id=spec.get("_job_id"),
                fire_webhook=False,
            )
        )

    def store_job(self, job_id: str, info: Dict[str, Any]) -> None:
        with self.jobs_lock:
            self.jobs[job_id] = info
            while len(self.jobs) > self.max_jobs:
                _, old = self.jobs.popitem(last=False)
                old.pop("layers", None)
                for key in ("before_path", "result_path", "preview_path"):
                    p = old.get(key)
                    if p and Path(p).is_file():
                        try:
                            Path(p).unlink()
                        except Exception:
                            pass

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        if not SAFE_ID_RE.match(job_id):
            return None
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def cleanup(self) -> None:
        self.async_jobs.shutdown()
        with self.watch_lock:
            if self.watcher is not None:
                self.watcher.stop()
                self.watcher = None
        self.engine_pool.clear()
        self.temp_dir.cleanup()


def _build_api_config(state: WebUIState, body: Dict[str, Any]) -> EnhancementConfig:
    base = EnhancementConfig(models_dir=state.models_dir, quiet=True)
    prof_name = body.get("profile")
    if isinstance(prof_name, str) and prof_name.strip():
        base = apply_profile_to_config(prof_name.strip(), base)

    scale_raw = body.get("scale")
    id_guard_raw = body.get("identity_guard")
    creativity_raw = body.get("creativity")

    overrides: Dict[str, Any] = {}
    if "preset" in body and body.get("preset") not in (None, ""):
        overrides["preset"] = str(body["preset"])
    if scale_raw not in (None, ""):
        overrides["scale"] = float(scale_raw)
    if "tta" in body and body.get("tta") not in (None, ""):
        overrides["tta"] = int(body["tta"])
    if "deblur" in body and body.get("deblur") not in (None, ""):
        overrides["deblur"] = str(body["deblur"])
    if "face_restore" in body and body.get("face_restore") not in (None, ""):
        overrides["face_restore"] = str(body["face_restore"])
    if id_guard_raw not in (None, ""):
        overrides["identity_guard"] = max(0.0, min(1.0, float(id_guard_raw)))
    if "backend" in body and body.get("backend") not in (None, ""):
        overrides["backend"] = str(body["backend"])
    if creativity_raw not in (None, ""):
        overrides["creativity"] = max(0.0, min(1.0, float(creativity_raw)))
    if "bit_depth" in body and body.get("bit_depth") not in (None, ""):
        overrides["bit_depth"] = int(body["bit_depth"])

    return replace(base, models_dir=state.models_dir, quiet=True, **overrides)


def _process_enhance_request(
    state: WebUIState,
    body: Dict[str, Any],
    *,
    fixed_job_id: Optional[str] = None,
    fire_webhook: bool = True,
) -> Dict[str, Any]:
    raw_b64 = str(body.get("data_base64") or "")
    if not raw_b64:
        raise ValueError("Missing data_base64 payload")
    img_bytes = _decode_base64_image(raw_b64)
    orig_name = Path(str(body.get("filename") or "image.png")).name
    src_ext = Path(orig_name).suffix.lower()
    if src_ext not in INPUT_EXTENSIONS:
        src_ext = ".png"

    out_ext = "." + str(body.get("output_ext") or "png").lstrip(".").lower()
    if out_ext not in SUPPORTED_EXTENSIONS:
        out_ext = ".png"

    job_id = fixed_job_id or uuid.uuid4().hex[:16]
    in_path = state.temp_root / f"{job_id}_in{src_ext}"
    before_png = state.temp_root / f"{job_id}_before.png"
    out_path = state.temp_root / f"{job_id}_out{out_ext}"
    preview_png = state.temp_root / f"{job_id}_preview.png"
    in_path.write_bytes(img_bytes)

    try:
        loaded = load_image(in_path)
    finally:
        in_path.unlink(missing_ok=True)
    save_image(before_png, loaded.rgb, alpha=loaded.alpha, bit_depth=8, dither=False)

    cfg = _build_api_config(state, body)
    with state.infer_lock:
        layers = prepare_enhancement_layers(
            loaded,
            cfg=cfg,
            engine_pool=state.engine_pool,
        )
        res = compose_enhancement_result(layers, cfg=cfg, include_neural_time=True)

    prev_saved = _write_job_outputs(
        out_path,
        preview_png,
        out_ext,
        res,
        cfg,
        loaded.exif_bytes,
        loaded.icc_profile,
    )

    dl_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(orig_name).stem) or "image"
    download_name = f"{dl_stem}_azzani{out_ext}"
    state.store_job(
        job_id,
        {
            "before_path": str(before_png),
            "result_path": str(out_path),
            "preview_path": str(prev_saved),
            "download_name": download_name,
            "dl_stem": dl_stem,
            "out_ext": out_ext,
            "layers": layers,
            "base_cfg": cfg,
            "exif_bytes": loaded.exif_bytes,
            "icc_profile": loaded.icc_profile,
            "rev": 1,
            "lock": threading.Lock(),
        },
    )
    response_payload: Dict[str, Any] = {
        "job_id": job_id,
        "before_url": f"/api/before/{job_id}",
        "preview_url": f"/api/preview/{job_id}?v=1",
        "result_url": f"/api/result/{job_id}?v=1",
        "download_name": download_name,
        "summary": res.summary_dict(),
        "histogram": _compute_histogram(res.output_rgb),
        "retune_defaults": _build_retune_defaults(layers, cfg, out_ext),
    }
    if fire_webhook:
        w_url = body.get("webhook_url")
        if isinstance(w_url, str) and w_url.strip():
            ok, _code, w_err = dispatch_webhook(
                w_url,
                "enhancement.completed",
                {
                    "job_id": job_id,
                    "download_name": download_name,
                    "result_url": response_payload["result_url"],
                    "summary": response_payload["summary"],
                },
                secret=body.get("webhook_secret"),
            )
            response_payload["webhook_delivered"] = ok
            if w_err:
                response_payload["webhook_error"] = w_err
    return response_payload


def _decode_base64_image(raw_b64: str) -> bytes:
    if "," in raw_b64 and raw_b64.lstrip().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    return base64.b64decode(raw_b64, validate=True)


def _make_handler(state: WebUIState):
    class AzzaniHandler(BaseHTTPRequestHandler):
        server_version = f"AzzaniEnhance/{__version__}"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # Keep terminal clean unless debugging.

        def _send_json(self, code: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length_hdr = self.headers.get("Content-Length", "0")
            try:
                length = int(length_hdr)
            except ValueError as err:
                raise ValueError("Invalid Content-Length header") from err
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                raise OverflowError(f"Payload exceeds {MAX_BODY_BYTES // (1024 * 1024)} MB limit")
            raw = self.rfile.read(length)
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("Expected JSON object")
            return obj

        def _serve_file(self, path: Path, *, download_name: Optional[str] = None) -> None:
            if not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "File not found"})
                return
            data = path.read_bytes()
            ctype, _ = mimetypes.guess_type(str(path))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if download_name:
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", download_name)
                self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
            self.end_headers()
            self.wfile.write(data)

        def _require_api_auth(self, path: str) -> bool:
            if not state.api_key or not path.startswith("/api/"):
                return True
            hdr_key = self.headers.get("X-API-Key", "")
            auth = self.headers.get("Authorization", "")
            token = auth[7:].strip() if auth.startswith("Bearer ") else hdr_key.strip()
            if token and hmac.compare_digest(token, state.api_key):
                return True
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "Missing or invalid API key"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if not self._require_api_auth(path):
                return

            if path == "/" or path == "/index.html":
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/healthz":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "version": __version__,
                        "queue_depth": state.async_jobs.queue_depth(),
                    },
                )
                return

            if path == "/api/doctor":
                self._send_json(HTTPStatus.OK, run_doctor())
                return

            if path == "/api/profiles":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "profiles": [
                            {"name": n, "description": d, "source": s}
                            for n, d, s in list_profiles()
                        ]
                    },
                )
                return

            if path.startswith("/api/jobs/"):
                job_id = path.split("/api/jobs/", 1)[1]
                if not SAFE_ID_RE.match(job_id):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid job_id"})
                    return
                info = state.async_jobs.get_public_status(job_id)
                if not info:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Async job not found"})
                    return
                self._send_json(HTTPStatus.OK, info)
                return

            if path == "/api/presets":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "presets": ["auto", *PRESET_MODEL_MAP.keys()],
                        "models": {k: v.purpose for k, v in MODEL_REGISTRY.items()},
                    },
                )
                return

            if path == "/api/watch/status":
                with state.watch_lock:
                    if state.watcher is None:
                        self._send_json(HTTPStatus.OK, {"running": False, "processed_count": 0, "error_count": 0})
                    else:
                        self._send_json(HTTPStatus.OK, state.watcher.status_dict())
                return

            if path.startswith("/api/before/"):
                job_id = path.split("/api/before/", 1)[1]
                job = state.get_job(job_id)
                if not job:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                    return
                self._serve_file(Path(job["before_path"]))
                return

            if path.startswith("/api/preview/"):
                job_id = path.split("/api/preview/", 1)[1]
                job = state.get_job(job_id)
                if not job:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                    return
                self._serve_file(Path(job.get("preview_path") or job["result_path"]))
                return

            if path.startswith("/api/result/"):
                job_id = path.split("/api/result/", 1)[1]
                job = state.get_job(job_id)
                if not job:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                    return
                dl = job["download_name"] if qs.get("download") == ["1"] else None
                self._serve_file(Path(job["result_path"]), download_name=dl)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if not self._require_api_auth(path):
                return
            try:
                body = self._read_json_body()
            except OverflowError as err:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(err)})
                return
            except Exception as err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                return

            if path == "/api/enhance":
                try:
                    self._send_json(HTTPStatus.OK, _process_enhance_request(state, body))
                except Exception as err:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                return

            if path == "/api/enhance/async":
                try:
                    raw_b64 = str(body.get("data_base64") or "")
                    if not raw_b64:
                        raise ValueError("Missing data_base64 payload")
                    job_id = uuid.uuid4().hex[:16]
                    spec = dict(body)
                    spec["_job_id"] = job_id
                    status = state.async_jobs.enqueue(job_id, spec)
                    status["status_url"] = f"/api/jobs/{job_id}"
                    self._send_json(HTTPStatus.ACCEPTED, status)
                except OverflowError as err:
                    self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(err)})
                except Exception as err:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                return

            if path == "/api/retune":
                try:
                    job_id = str(body.get("job_id") or "")
                    job = state.get_job(job_id)
                    if not job:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Job not found"})
                        return

                    def _clamp_opt(name: str, lo: float, hi: float, fallback: Optional[float]) -> Optional[float]:
                        val = body.get(name)
                        if val in (None, ""):
                            return fallback
                        return max(lo, min(hi, float(val)))

                    base_cfg: EnhancementConfig = job["base_cfg"]
                    wb_raw = _clamp_opt("wb_strength", 0.0, 1.0, base_cfg.white_balance_strength)
                    out_ext = "." + str(body.get("output_ext") or job["out_ext"]).lstrip(".").lower()
                    if out_ext not in SUPPORTED_EXTENSIONS:
                        out_ext = job["out_ext"]

                    retune_cfg = replace(
                        base_cfg,
                        tone_strength=_clamp_opt("tone", 0.0, 1.0, base_cfg.tone_strength),
                        detail_strength=_clamp_opt("detail", 0.0, 1.5, base_cfg.detail_strength),
                        color_fix=float(_clamp_opt("color_fix", 0.0, 1.0, base_cfg.color_fix) or 0.0),
                        white_balance=("on" if (wb_raw is not None and wb_raw > 0.0) else ("off" if wb_raw == 0.0 else base_cfg.white_balance)),
                        white_balance_strength=wb_raw,
                        ca_suppress=float(_clamp_opt("ca_suppress", 0.0, 1.0, base_cfg.ca_suppress) or 0.0),
                        halo_guard=float(_clamp_opt("halo_guard", 0.0, 1.0, base_cfg.halo_guard) or 0.0),
                        face_fidelity=float(_clamp_opt("face_fidelity", 0.0, 1.0, base_cfg.face_fidelity) or 0.0),
                        natural_grain=_clamp_opt("grain", 0.0, 0.02, base_cfg.natural_grain),
                        bokeh_guard=bool(body["bokeh_guard"]) if "bokeh_guard" in body else base_cfg.bokeh_guard,
                        bit_depth=int(body.get("bit_depth") or base_cfg.bit_depth),
                    )

                    with job["lock"]:
                        res = compose_enhancement_result(
                            job["layers"],
                            cfg=retune_cfg,
                            include_neural_time=False,
                        )
                        rev = int(job["rev"]) + 1
                        job["rev"] = rev
                        out_path = state.temp_root / f"{job_id}_out{out_ext}"
                        preview_png = state.temp_root / f"{job_id}_preview.png"
                        old_result = Path(job["result_path"])
                        if old_result != out_path and old_result.is_file():
                            old_result.unlink(missing_ok=True)
                        prev_saved = _write_job_outputs(
                            out_path,
                            preview_png,
                            out_ext,
                            res,
                            retune_cfg,
                            job.get("exif_bytes"),
                            job.get("icc_profile"),
                        )
                        dl_name = f"{job['dl_stem']}_azzani{out_ext}"
                        job["result_path"] = str(out_path)
                        job["preview_path"] = str(prev_saved)
                        job["out_ext"] = out_ext
                        job["download_name"] = dl_name

                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "job_id": job_id,
                            "before_url": f"/api/before/{job_id}",
                            "preview_url": f"/api/preview/{job_id}?v={rev}",
                            "result_url": f"/api/result/{job_id}?v={rev}",
                            "download_name": dl_name,
                            "summary": res.summary_dict(),
                            "histogram": _compute_histogram(res.output_rgb),
                            "retuned": True,
                        },
                    )
                except Exception as err:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                return

            if path == "/api/watch/start":
                try:
                    wdir = Path(str(body.get("watch_dir") or "./azzani_inbox")).expanduser().resolve()
                    odir = Path(str(body.get("output_dir") or str(wdir / "enhanced"))).expanduser().resolve()
                    w_cfg = EnhancementConfig(
                        preset=str(body.get("preset") or "auto"),
                        tta=int(body.get("tta") or 1),
                        models_dir=state.models_dir,
                        quiet=True,
                    )
                    with state.watch_lock:
                        if state.watcher is not None:
                            state.watcher.stop()
                        state.watcher = WatchFolderRunner(
                            wdir,
                            odir,
                            cfg=w_cfg,
                            engine_pool=state.engine_pool,
                        )
                        state.watcher.start_background()
                        self._send_json(HTTPStatus.OK, state.watcher.status_dict())
                except Exception as err:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                return

            if path == "/api/watch/stop":
                with state.watch_lock:
                    if state.watcher is not None:
                        state.watcher.stop()
                        st = state.watcher.status_dict()
                    else:
                        st = {"running": False}
                self._send_json(HTTPStatus.OK, st)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})

    return AzzaniHandler


def create_web_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    models_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> tuple[ThreadingHTTPServer, WebUIState]:
    """Create a ThreadingHTTPServer bound to host:port (port=0 picks an ephemeral free port)."""
    state = WebUIState(models_dir=models_dir, api_key=api_key)
    handler = _make_handler(state)
    server = ThreadingHTTPServer((host, int(port)), handler)
    return server, state


def _open_browser_quietly(url: str) -> None:
    """Open the local Web UI in the user's browser without polluting CLI stderr with Snap/Chromium logs."""
    try:
        if sys.platform.startswith("linux"):
            xdg = shutil.which("xdg-open")
            if xdg:
                subprocess.Popen(
                    [xdg, url],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return
        elif sys.platform == "darwin":
            opener = shutil.which("open")
            if opener:
                subprocess.Popen(
                    [opener, url],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return
        webbrowser.open(url)
    except Exception:
        pass


def launch_web_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    models_dir: Optional[Path] = None,
) -> int:
    """Start the local AZZANI Image Enhancements Web UI and block until Ctrl-C."""
    try:
        server, state = create_web_server(host=host, port=port, models_dir=models_dir)
    except OSError:
        # Port in use -> bind cleanly to any free ephemeral port so user never gets an error.
        server, state = create_web_server(host=host, port=0, models_dir=models_dir)

    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"AZZANI Image Enhancements Web UI running at: {url}")
    print("Press Ctrl+C to stop cleanly.")
    if open_browser:
        threading.Timer(0.35, lambda: _open_browser_quietly(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AZZANI Web UI cleanly...")
    finally:
        server.server_close()
        state.cleanup()
    return 0
