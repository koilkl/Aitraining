#!/usr/bin/env python3
"""Build the AIoScout design-system showcase poster.

Pipeline
--------
1. Load the real ``AIoScout_UI_Shell.html`` and inject a *capture-only* debug
   bridge inside its two IIFEs.  No markup, CSS, copy, or layout is changed —
   the bridge only exposes the shell's own state + render functions so the
   poster can seed a realistic classroom project and drive the real views.
2. Launch headless Chromium (Playwright), force the light editorial theme,
   load Inter / JetBrains Mono, seed a 3-plate SIGN_OCR classroom project, and
   capture the five product views at 3x device scale.
3. Compose ``AIoScout_Design_Poster.html`` (self-contained: fonts + captures
   embedded as data URLs) in the same editorial-tech instrument language.
4. Render the poster to PNG (3600 x 5100) and a print-sized A2 PDF (420 x 594 mm).

Deliverables (AItraining/docs/):
    AIoScout_Design_Poster.html
    AIoScout_Design_Poster.png
    AIoScout_Design_Poster.pdf

Run:
    AItraining/.venv/bin/python AItraining/tools/build_design_poster.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: AItraining/.venv/bin/pip install pillow")

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sys.exit("Playwright is required: AItraining/.venv/bin/pip install playwright")

REPO = Path(__file__).resolve().parents[2]
SHELL = REPO / "AIoScout_UI_Shell.html"
DOCS = REPO / "AItraining" / "docs"
FONTS = DOCS / "fonts"
BUILD = DOCS / ".poster_build"

POSTER_CSS_W = 1200
POSTER_CSS_H = 1700
CAPTURE_W = 1600
CAPTURE_H = 1000
DSF = 3

# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

FONT_FILES = {
    ("Inter", 400): "Inter-400.woff2",
    ("Inter", 600): "Inter-600.woff2",
    ("Inter", 700): "Inter-700.woff2",
    ("Inter", 800): "Inter-800.woff2",
    ("JetBrains Mono", 400): "JetBrainsMono-400.woff2",
    ("JetBrains Mono", 600): "JetBrainsMono-600.woff2",
    ("JetBrains Mono", 700): "JetBrainsMono-700.woff2",
}


GF_CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap"
)
GF_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def ensure_fonts() -> None:
    """Download the latin woff2 subsets once, then reuse the cached files."""
    FONTS.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FONT_FILES.values() if not (FONTS / name).exists()]
    if missing:
        req = urllib.request.Request(GF_CSS_URL, headers={"User-Agent": GF_UA})
        with urllib.request.urlopen(req, timeout=30) as res:
            css = res.read().decode("utf-8")
        blocks = re.findall(r"/\* latin \*/\s*@font-face\s*\{(.*?)\}", css, re.S)
        urls = {}
        for block in blocks:
            family = re.search(r"font-family: '([^']+)'", block)
            weight = re.search(r"font-weight: (\d+)", block)
            url = re.search(r"url\((https://[^)]+)\)", block)
            if family and weight and url:
                urls[(family.group(1), int(weight.group(1)))] = url.group(1)
        for key, name in FONT_FILES.items():
            if (FONTS / name).exists():
                continue
            url = urls.get(key)
            if not url:
                raise RuntimeError(f"Google Fonts did not serve {key}")
            urllib.request.urlretrieve(url, FONTS / name)
        print(f"[0/4] fonts cached     {FONTS.relative_to(REPO)} ({len(missing)} downloaded)")
    for license_file in ("OFL-Inter.txt", "OFL-JetBrainsMono.txt"):
        if not (FONTS / license_file).exists():
            print(f"      warning: missing {license_file} (SIL OFL notice)")


def font_face_css() -> str:
    faces = []
    for (family, weight), filename in FONT_FILES.items():
        path = FONTS / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing font {path}. Re-download the latin woff2 subsets into AItraining/docs/fonts/."
            )
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        faces.append(
            "@font-face{font-family:'%s';font-style:normal;font-weight:%d;font-display:block;"
            "src:url(data:font/woff2;base64,%s) format('woff2');}" % (family, weight, b64)
        )
    return "".join(faces)


# ---------------------------------------------------------------------------
# shell bridge injection (capture-only; visuals untouched)
# ---------------------------------------------------------------------------

BRIDGE_MAIN = """
  /* ---- build_design_poster.py capture bridge (debug only) ---- */
  window.__POSTER_CAP__ = {
    state, openProject, ensureProject, createProject, switchPage, renderRecent,
    renderWorkspace, renderCollectPage, renderEditPage, renderEditToc,
    renderEditWorkbench, loadEditDraft, renderPreviewPage, renderPreviewStage,
    applyPreviewPrediction, applyPreviewRoi, applyPreviewSign,
    updatePreviewHUD, buildPreviewOutputs, renderTrainPage, syncSignClasses,
    signPlateScene, signPreviewScene, sampleItemsForClass, toast
  };
"""

BRIDGE_PATCH = """
  /* ---- build_design_poster.py train-patch bridge (debug only) ---- */
  window.__POSTER_TPATCH__ = {
    set(overrides) { Object.assign(patch, overrides); },
    get() { return patch; },
    render: renderAll,
    ensure: ensurePatch
  };
"""


def make_capture_shell() -> Path:
    html = SHELL.read_text(encoding="utf-8")
    main_marker = "\n})();\n</script>\n<script id=\"tr-train-patch-js\">"
    if html.count(main_marker) != 1:
        raise RuntimeError("Could not locate the main IIFE boundary in AIoScout_UI_Shell.html")
    html = html.replace(main_marker, "\n" + BRIDGE_MAIN + main_marker, 1)

    tail = "\n})();\n\n</script>\n</body>"
    if html.count(tail) != 1:
        raise RuntimeError("Could not locate the train-patch IIFE boundary in AIoScout_UI_Shell.html")
    html = html.replace(tail, "\n" + BRIDGE_PATCH + tail, 1)

    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "shell.capture.html"
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# deterministic classroom dataset (mirrors a real 3-plate sign project)
# ---------------------------------------------------------------------------

PLATES = ["A12", "B34", "C56"]
CLASS_COUNTS = {"A": 64, "1": 58, "2": 55, "B": 52, "3": 49, "4": 47, "C": 44, "5": 41, "6": 38}
CLASS_ORDER = ["A", "1", "2", "B", "3", "4", "C", "5", "6"]

SEED_LAUNCHER = """
() => {
  const c = window.__POSTER_CAP__;
  const row = (type, name, folder, time, accuracy, trained) => c.ensureProject({
    id: 'p_' + type + '_' + name.length,
    name: name + '.tmproj',
    type,
    path: '~/Projects/AIoScout/' + folder + '/' + name + '.tmproj',
    folder: '~/Projects/AIoScout/' + folder + '/',
    time,
    accuracy: trained ? accuracy : 0,
    trained: !!trained,
    classes: [
      { id: 'x1', name: 'A', samples: 64 },
      { id: 'x2', name: '1', samples: 58 },
      { id: 'x3', name: '2', samples: 55 }
    ]
  });
  c.state.projects = [
    row('car', 'sign_A12_classroom', 'sign_A12_classroom', '14:32', 0.967, true),
    row('ocr', 'ocr_gate_numbers', 'ocr_gate_numbers', 'Yesterday', 0.941, true),
    row('image', 'traffic_cone_colors', 'traffic_cone_colors', 'Nov 18', 0.902, true),
    row('car', 'plate_whitelist_v3', 'plate_whitelist_v3', 'Nov 12', 0, false),
    row('ocr', 'ocr_math_symbols', 'ocr_math_symbols', 'Nov 09', 0.913, true),
    row('image', 'classroom_objects', 'classroom_objects', 'Nov 04', 0.884, true),
    row('car', 'gate_plate_mock', 'gate_plate_mock', 'Oct 30', 0, false),
    row('image', 'corridor_states', 'corridor_states', 'Oct 27', 0.861, true)
  ];
  c.renderRecent();
}
"""

SEED_PROJECT = """
() => {
  const c = window.__POSTER_CAP__;
  const proj = c.ensureProject({
    id: 'project_sign_a12',
    name: 'sign_A12_classroom.tmproj',
    type: 'car',
    path: '~/Projects/AIoScout/sign_A12_classroom/sign_A12_classroom.tmproj',
    folder: '~/Projects/AIoScout/sign_A12_classroom/',
    time: '14:32',
    accuracy: 0.967,
    trained: true,
    finalLoss: 0.041,
    size: '1.18 MB',
    latency: '8.42 ms',
    licenses: [{ id: 'l1', plate: 'A12' }, { id: 'l2', plate: 'B34' }, { id: 'l3', plate: 'C56' }],
    classes: []
  });
  proj.licenses = [{ id: 'l1', plate: 'A12' }, { id: 'l2', plate: 'B34' }, { id: 'l3', plate: 'C56' }];
  c.syncSignClasses(proj, { force: true });
  const counts = %s;
  proj.classes.forEach((cls) => { cls.samples = counts[cls.name] || 40; });
  proj.trainHistory = %s;
  proj.settings.reading_order = 'ltr';
  proj.settings.ood = { maxProb: 0.40, entropy: 0.85, signMin: 0.3, signMax: 70 };
  c.state.projects = [proj, ...c.state.projects];
  c.openProject(proj);
  return proj.name;
}
""" % (
    json.dumps(CLASS_COUNTS),
    json.dumps(
        {
            "acc": [],  # filled client-side below
            "loss": [],
        }
    ),
)

# 28-epoch deterministic history: slow start, clean convergence.
HISTORY_JS = """
() => {
  const c = window.__POSTER_CAP__;
  const proj = c.state.project;
  const points = 120;
  const targetAcc = 0.967;
  const targetLoss = 0.041;
  const startAcc = 0.582;
  const startLoss = 1.842;
  const tau = 30;
  const norm = 1 - Math.exp(-(points - 1) / tau);
  const acc = [], loss = [], history = [];
  for (let i = 0; i < points; i += 1) {
    const t = i / (points - 1);
    // Exponential convergence on the epoch axis, normalised so epoch 120
    // lands exactly on the KPI values; the wobble fades at the final epoch.
    const a = startAcc + (targetAcc - startAcc) * ((1 - Math.exp(-i / tau)) / norm)
      - 0.0035 * Math.sin(i * 1.9) * (1 - t);
    const l = targetLoss + (startLoss - targetLoss) * ((Math.exp(-i / tau) - Math.exp(-(points - 1) / tau)) / norm)
      + 0.006 * Math.cos(i * 1.7) * (1 - t);
    acc.push(Math.min(0.995, Math.max(0.35, a)));
    loss.push(Math.max(0.028, l));
    history.push({ epoch: i + 1, acc: acc[i], val_accuracy: acc[i], loss: loss[i], val_loss: loss[i] });
  }
  // Last point is exact (the wobble vanishes and the normalisation is 1).
  history[points - 1].val_accuracy = targetAcc;
  history[points - 1].acc = targetAcc;
  history[points - 1].val_loss = targetLoss;
  history[points - 1].loss = targetLoss;
  acc[points - 1] = targetAcc;
  loss[points - 1] = targetLoss;
  proj.trainHistory = { acc: acc.map((v) => v * 100), loss };
  c.state.trainChart = { acc: acc.map((v) => v * 100), loss: loss.slice() };
  return history;
}
"""

SET_EDIT_MANUAL_JS = """
() => {
  const c = window.__POSTER_CAP__;
  const proj = c.state.project;
  const cls = proj.classes.find((item) => item.name === 'A') || proj.classes[0];
  c.state.ui.selectedClassId = cls.id;
  cls.preprocess = { mode: 'manual_roi', dark: 0, lum: 100, roi: { x: 0.22, y: 0.19, w: 0.56, h: 0.62 } };
  cls.samplePreprocess = {};
  c.state.ui.editSampleIndex = 0;
  c.state.ui.editSelection = [];
  c.state.ui.editMulti = false;
  c.state.ui.editView = 'raw';
  c.state.ui.editDraft = null;
  return cls.name;
}
"""

SEED_TRAIN_JS = """
(history) => {
  const c = window.__POSTER_CAP__;
  const t = window.__POSTER_TPATCH__;
  t.ensure();
  const consoleLines = ['$ aioscout train --int8 --device esp32-p4'];
  consoleLines.push('dataset · 9 classes · 448 samples · tensor 3 × 28 × 28 × 1 GRAY');
  consoleLines.push('epochs 120 · batch 32 · val 0.25 · preprocess BT.601 → contrast → INT8');
  const milestones = new Set([0, 39, 79, 119]);
  history.forEach((h, i) => {
    if (milestones.has(i)) {
      consoleLines.push(
        'epoch ' + String(h.epoch).padStart(3, '0') +
        ' · loss ' + h.loss.toFixed(4) +
        ' · val_acc ' + (h.val_accuracy * 100).toFixed(2) + String.fromCharCode(37)
      );
    }
  });
  consoleLines.push('tflite · INT8 flatbuffer 1.18 MB · P4 latency 8.42 ms');
  consoleLines.push('done · val_acc 96.7%% · runs/latest/ · snapshot ./backup/');
  t.set({
    type: 'sign_ocr',
    trained: true,
    dirty: false,
    running: false,
    accuracy: 96.7,
    loss: 0.041,
    sizeBytes: 1236000,
    latencyMs: 8.42,
    history,
    chart: { acc: history.map((h) => h.val_accuracy * 100), loss: history.map((h) => h.val_loss) },
    matrixData: {
      labels: %s,
      matrix: %s
    },
    console: consoleLines
  });
  t.render();
}
""" % (
    json.dumps(CLASS_ORDER),
    json.dumps(
        [
            [60, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 56, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 53, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 50, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 48, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 46, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 43, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 40, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 37],
        ]
    ),
)

# Preview: use the shell's own simulated plate scene as the inference frame,
# rasterised 480x360 (4:3, "slice" like the live stage), then replay the real
# applyPreviewPrediction path with real per-slot boxes.
PREVIEW_JS = """
async () => {
  const c = window.__POSTER_CAP__;
  const proj = c.state.project;
  c.buildPreviewOutputs();
  const sceneHost = document.getElementById('stageScene');
  const svg = sceneHost.querySelector('svg').cloneNode(true);
  svg.setAttribute('width', '480');
  svg.setAttribute('height', '480');
  const svgText = new XMLSerializer().serializeToString(svg);
  const frameB64 = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const square = document.createElement('canvas');
      square.width = 480; square.height = 480;
      const sctx = square.getContext('2d');
      sctx.drawImage(img, 0, 0, 480, 480);
      // 4:3 stage: the shell's SVG uses preserveAspectRatio="slice", which
      // crops 60 px top + bottom of the 480×480 scene into the 4:3 frame.
      const canvas = document.createElement('canvas');
      canvas.width = 480; canvas.height = 360;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(square, 0, -60, 480, 480);
      resolve(canvas.toDataURL('image/png').split(',')[1]);
    };
    img.onerror = reject;
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgText)));
  });

  sceneHost.innerHTML = '<img id="previewFrameImg" alt="Inference frame">';
  const frame = document.getElementById('previewFrameImg');
  c.state.previewFrameEl = frame;
  frame.src = 'data:image/png;base64,' + frameB64;
  await frame.decode();

  const preview = c.state.ui.preview;
  preview.source = 'webcam';
  preview.roiOn = false;
  preview.origOn = false;
  preview.inputOn = true;
  preview.synth = 'match';
  c.updatePreviewHUD();

  const slot = (x1, x2, ch, prob) => ({
    char: ch,
    label: ch,
    prob,
    box_norm: [x1, 0.5444, x2, 0.7389],
    match: true,
    ood: false
  });
  const pred = {
    match: true,
    ood: false,
    top_label: 'A12',
    top_prob: 0.9977,
    max_prob: 0.9977,
    entropy_ratio: 0.031,
    ood_reason: 'in distribution',
    match_text: 'whitelist match · 3/3 slots',
    reading_order: 'ltr',
    sign_sequence: ['A', '1', '2'],
    model_source: 'latest',
    stage_image_b64: frameB64,
    slots: [
      slot(0.2167, 0.3917, 'A', 0.9981),
      slot(0.4125, 0.5875, '1', 0.9974),
      slot(0.6083, 0.7833, '2', 0.9977)
    ]
  };
  c.applyPreviewPrediction(pred);
  await frame.decode();
  c.applyPreviewRoi(pred);
  c.applyPreviewSign(pred, proj);
  const status = document.getElementById('previewStageStatus');
  if (status) status.hidden = true;
  return { frame: [frame.naturalWidth, frame.naturalHeight] };
}
"""


# ---------------------------------------------------------------------------
# browser capture
# ---------------------------------------------------------------------------

def hide_chrome(page):
    page.evaluate("() => { document.activeElement && document.activeElement.blur && document.activeElement.blur(); }")
    page.mouse.move(CAPTURE_W - 6, CAPTURE_H - 6)
    page.wait_for_timeout(450)


def assert_ui_state(page, stage: str):
    """Fail loudly if the seeded classroom state did not actually render."""
    checks = {
        "launcher": """
            () => {
              const visible = (el) => el && getComputedStyle(el).display !== 'none';
              return {
                recent: document.querySelectorAll('.recent-row').length,
                tooltip: [...document.querySelectorAll('.ui-tooltip')].filter(visible).length,
                toast: document.querySelectorAll('.toast').length
              };
            }
        """,
        "collect": """
            () => {
              const imgs = document.querySelectorAll('#recentStrip img, .cl-samples-grid img, .recent-thumb img').length;
              return {
                classes: document.querySelectorAll('#classBoard .class-card').length,
                plates: document.querySelectorAll('#plateWhitelistDeck .plate-chip').length,
                thumbs: imgs,
                stage: !!document.querySelector('#studioBody .feed-stage svg'),
                toast: document.querySelectorAll('.toast').length
              };
            }
        """,
        "edit": """
            () => {
              const txt = (id) => (document.getElementById(id)?.textContent || '').trim();
              const handles = [...document.querySelectorAll('#editRoi .roi-handle')]
                .filter((el) => getComputedStyle(el).display !== 'none').length;
              return {
                crop: txt('edCropPx'),
                otsu: txt('edOtsuThr'),
                meanLum: txt('edMeanLum'),
                tensor: txt('edTensorReadout'),
                handles,
                readout: !!document.getElementById('edReadout'),
                toast: document.querySelectorAll('.toast').length
              };
            }
        """,
        "train": """
            () => {
              const txt = (id) => (document.getElementById(id)?.textContent || '').trim();
              return {
                acc: txt('kpiAcc'),
                loss: txt('kpiLoss'),
                flash: txt('kpiFlash'),
                latency: txt('kpiLatency'),
                cells: document.querySelectorAll('#trainConfusion .tr-conf-cell').length,
                accPoints: (document.getElementById('accLine')?.getAttribute('points') || '').trim().length,
                lossPoints: (document.getElementById('lossLine')?.getAttribute('points') || '').trim().length,
                console: (document.getElementById('trainConsole')?.textContent || ''),
                toast: document.querySelectorAll('.toast').length
              };
            }
        """,
        "preview": """
            () => {
              const txt = (id) => (document.getElementById(id)?.textContent || '').trim();
              return {
                verdict: txt('resultVerdict'),
                klass: txt('resultClass'),
                conf: txt('resultConf'),
                slots: document.querySelectorAll('#signSlots .signmatch-slot').length,
                faders: document.querySelectorAll('.pv-fader').length,
                roiVisible: !document.getElementById('roiOverlay')?.hidden,
                slotBoxes: document.querySelectorAll('#roiSlotLayer .roi-slot').length,
                model: txt('modelSourceChip'),
                exportCard: !!document.querySelector('#pagePreview .export-card'),
                toast: document.querySelectorAll('.toast').length
              };
            }
        """,
    }
    state = page.evaluate(checks[stage])
    ok = True
    if stage == "launcher":
        ok = state["recent"] == 8 and state["tooltip"] == 0 and state["toast"] == 0
    elif stage == "collect":
        ok = state["classes"] == 9 and state["plates"] == 3 and state["thumbs"] >= 20 and state["stage"] and state["toast"] == 0
    elif stage == "edit":
        ok = (state["handles"] == 8 and state["readout"] and state["crop"] not in ("", "—")
              and state["otsu"] not in ("", "—") and state["tensor"] and state["toast"] == 0)
    elif stage == "train":
        ok = (state["acc"] == "96.7%" and state["loss"] == "0.041" and state["flash"] == "1.18 MB"
              and state["latency"] == "8.42 ms" and state["cells"] >= 81
              and state["accPoints"] > 100 and state["lossPoints"] > 100
              and "done · val_acc 96.7%" in state["console"] and state["toast"] == 0)
    elif stage == "preview":
        ok = ("YES" in state["verdict"] and "A12" in state["klass"] and state["slots"] == 3
              and state["faders"] == 6 and state["roiVisible"] and state["slotBoxes"] == 3
              and "LATEST" in state["model"] and state["exportCard"] and state["toast"] == 0)
    if not ok:
        raise RuntimeError(f"Capture state failed for {stage}: {json.dumps(state, ensure_ascii=False)}")
    return state


def capture_views(shell_path: Path) -> dict:
    BUILD.mkdir(parents=True, exist_ok=True)
    result = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-color-profile=srgb", "--font-render-hinting=none"])
        context = browser.new_context(
            viewport={"width": CAPTURE_W, "height": CAPTURE_H},
            device_scale_factor=DSF,
            color_scheme="light",
            reduced_motion="reduce",
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(shell_path.as_uri())
        page.wait_for_timeout(400)
        page.add_style_tag(content=font_face_css())
        page.add_style_tag(content='.ui-tooltip{display:none!important}')
        page.evaluate("() => document.fonts.ready")
        page.evaluate("() => document.documentElement.setAttribute('data-theme','light')")
        page.wait_for_timeout(250)

        # ---- Shot A — launcher -------------------------------------------------
        page.evaluate(SEED_LAUNCHER)
        hide_chrome(page)
        assert_ui_state(page, "launcher")
        result["launcher"] = page.screenshot(path=str(BUILD / "shot_a_launcher.png"))

        # ---- Shot B — collect --------------------------------------------------
        page.evaluate(SEED_PROJECT)
        page.wait_for_timeout(700)
        hide_chrome(page)
        assert_ui_state(page, "collect")
        result["collect"] = page.screenshot(path=str(BUILD / "shot_b_collect.png"))

        # ---- Shot C — edit (manual ROI, 8 handles) ------------------------------
        # Set the class contract BEFORE navigating: the patched edit renderer
        # (installed on DOMContentLoaded) reads cls.preprocess through the live
        # page mount.  Calling the bridge's stale render helpers afterwards
        # would overwrite the patched inspector (and drop the Otsu card).
        page.evaluate(SET_EDIT_MANUAL_JS)
        page.evaluate("() => window.__POSTER_CAP__.switchPage('edit')")
        page.wait_for_timeout(650)
        readouts = page.evaluate(
            """
            () => {
              const t = (id) => { const el = document.getElementById(id); return el ? el.textContent.trim() : ''; };
              return { crop: t('edCropPx'), otsu: t('edOtsuThr'), meanLum: t('edMeanLum'), tensor: t('edTensorReadout') };
            }
            """
        )
        hide_chrome(page)
        assert_ui_state(page, "edit")
        result["edit"] = page.screenshot(path=str(BUILD / "shot_c_edit.png"))
        edit_rects = page.evaluate(
            """
            () => {
              const r = (sel) => {
                const el = document.querySelector(sel);
                if (!el) return null;
                const b = el.getBoundingClientRect();
                return { x: b.x, y: b.y, w: b.width, h: b.height };
              };
              return {
                editStage: r('#editOriginalStage'),
                editRoi: r('#editRoi'),
                editInspector: r('#editWorkbench .edit-inspector')
              };
            }
            """
        )

        # ---- Shot D — train ----------------------------------------------------
        page.evaluate("() => window.__POSTER_CAP__.switchPage('train')")
        page.wait_for_timeout(450)
        history = page.evaluate(HISTORY_JS)
        page.evaluate(SEED_TRAIN_JS, history)
        page.wait_for_timeout(350)
        hide_chrome(page)
        assert_ui_state(page, "train")
        result["train"] = page.screenshot(path=str(BUILD / "shot_d_train.png"))

        # ---- Shot E — preview / export -----------------------------------------
        page.evaluate("() => window.__POSTER_CAP__.switchPage('preview')")
        page.wait_for_timeout(500)
        frame_info = page.evaluate(PREVIEW_JS)
        page.wait_for_timeout(450)
        hide_chrome(page)

        # detail crops (real DOM regions, captured at the same 3x scale)
        preview_rects = page.evaluate(
            """
            () => {
              const r = (sel) => {
                const el = document.querySelector(sel);
                if (!el) return null;
                const b = el.getBoundingClientRect();
                return { x: b.x, y: b.y, w: b.width, h: b.height };
              };
              return {
                stage: r('#previewStage'),
                thresholds: r('.pv-thresholds-panel'),
                faderBank: r('.pv-fader-bank'),
                exportCard: r('#pagePreview .export-card')
              };
            }
            """
        )
        assert_ui_state(page, "preview")
        result["preview"] = page.screenshot(path=str(BUILD / "shot_e_preview.png"))
        context.close()
        browser.close()

    if errors:
        raise RuntimeError("Shell raised JS errors during capture:\n  " + "\n  ".join(errors))

    detail_rects = dict(edit_rects or {})
    detail_rects.update(preview_rects or {})
    for key in ("editStage", "editRoi", "thresholds", "faderBank"):
        if not detail_rects.get(key):
            raise RuntimeError(f"Missing capture rect: {key}")
    (BUILD / "detail_rects.json").write_text(json.dumps({"frame": frame_info, "rects": detail_rects, "readouts": readouts}, indent=2))
    return result


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def png_bytes(path_or_bytes, max_w: int) -> bytes:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    else:
        im = Image.open(path_or_bytes)
    im = im.convert("RGB")
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    return out.getvalue()


def crop_rect(shot_png: Path, rect: dict, pad: int = 0) -> Image.Image:
    im = Image.open(shot_png).convert("RGB")
    x = max(0, int((rect["x"] - pad) * DSF))
    y = max(0, int((rect["y"] - pad) * DSF))
    x2 = min(im.width, int((rect["x"] + rect["w"] + pad) * DSF))
    y2 = min(im.height, int((rect["y"] + rect["h"] + pad) * DSF))
    return im.crop((x, y, x2, y2))


def crop_bytes(shot_png: Path, rect: dict, max_w: int, pad: int = 0) -> str:
    im = crop_rect(shot_png, rect, pad)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def data_url_for(shot_png: Path, max_w: int, crop: dict | None = None) -> str:
    if crop:
        return crop_bytes(shot_png, crop, max_w)
    raw = png_bytes(shot_png, max_w)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# poster HTML
# ---------------------------------------------------------------------------

LOGO_PATH = (
    "M405.08 0H90C40.3 0 0 40.3 0 90V405.07C0 454.78 40.3 495.07 90 495.07H405.07C454"
    ".78 495.07 495.07 454.77 495.07 405.07V90C495.07 40.29 454.77 0 405.07 0H405.08Z"
    "M123.27 376.8L71.73 428.34C68.15 431.92 62.35 431.92 58.77 428.34L52.94 422.51C4"
    "9.36 418.93 49.36 413.13 52.94 409.55L104.48 358.01C108.06 354.43 113.86 354.43 "
    "117.44 358.01L123.27 363.84C126.85 367.42 126.85 373.22 123.27 376.8ZM333.43 339"
    ".2C299.8 383.22 241.33 404.16 184.14 386.85C141.32 373.89 107.18 339.56 94.09 29"
    "6.78C68.1 211.82 127.9 127.92 211.74 119.91C258.02 115.48 303.39 135.01 332.08 1"
    "71.26C334.42 174.22 333.69 178.54 330.62 180.73L319.95 188.33C317.06 190.39 313."
    "01 189.7 310.78 186.93C284.58 154.43 240.46 137.51 195.13 149.9C151.62 161.79 11"
    "9.21 200.5 115.21 245.42C108.91 316.34 168.12 372.24 235.49 365.8C266.82 362.8 2"
    "93.89 346.91 311.82 323.86C314 321.05 318.04 320.29 320.97 322.3L331.82 329.76C3"
    "34.92 331.89 335.72 336.19 333.44 339.19L333.43 339.2ZM161.83 279.24C170.03 302."
    "03 190.77 318.93 214.75 322.31C264.21 329.28 303.07 283.83 289.48 236.56C283.54 "
    "215.9 267.34 198.96 246.98 192.08C230.95 186.66 215.22 187.6 201.69 192.55C198.5"
    "2 193.71 194.86 192.45 193.41 189.41L188.24 178.56C185.95 173.75 187.43 169.58 1"
    "91 168.2C199.01 165.09 207.44 163.08 216.04 162.26C273.49 156.77 324.62 205.4 31"
    "8.27 265.97C314.41 302.86 288 334.8 252.48 345.49C194.29 363 137.01 322.05 131.5"
    "2 264.64C130.7 256.04 131.07 247.39 132.61 238.93C133.3 235.17 137.11 232.92 140"
    ".81 233.88C144.81 234.92 149.84 236.23 153.72 237.24C157.15 238.13 159.07 241.5 "
    "158.52 245C156.79 255.83 157.61 267.45 161.85 279.24H161.83ZM387.56 427.04C365.1"
    "4 422.51 349.66 401.8 349.66 378.93C349.66 309.15 349.66 109.74 349.66 70.73C349"
    ".66 67.05 352.64 64.06 356.33 64.06H369.57C373.25 64.06 376.24 67.04 376.24 70.7"
    "3V378.97C376.24 389.59 383.46 399.46 393.94 401.18C405.44 403.07 415.61 395.64 4"
    "18.15 385.25C418.83 382.46 421.84 380.19 424.71 380.19H438.13C442.28 380.19 445."
    "37 383.98 444.69 388.08C440.42 413.87 415.53 432.69 387.57 427.04H387.56Z"
)
LOGO_SVG = (
    '<svg class="mark" viewBox="0 0 496 496" aria-label="AIoScout"><path fill="currentColor" d="%s"/></svg>'
    % LOGO_PATH
)

SWATCHES = [
    ("--canvas", "#FFFFFF", "Pure Paper"),
    ("--canvas-soft", "#F9F9F8", "Drafting Ground"),
    ("--panel", "#F3F4F3", "Recessed Well"),
    ("--primary", "#000000", "Structural Ink"),
    ("--ink", "#1A1C1C", "Body Ink"),
    ("--mute", "#5E5E5E", "Annotation"),
    ("--accent-orange", "#F58331", "Active Focus · ROI"),
    ("--accent-purple", "#5B5BA8", "Eyebrow · OCR · Loss"),
    ("--success", "#1F8A4C", "In-Distribution"),
    ("--danger", "#B3261E", "OOD · Destructive"),
]


def swatch_html(name, hexv, role, dark_text=False):
    border = "#d8d9d8"
    return f"""
    <div class="sw">
      <span class="sw-chip" style="background:{hexv};border-color:{border}"></span>
      <span class="sw-meta"><b>{hexv}</b><i>{name}</i><em>{role}</em></span>
    </div>"""


def poster_html(assets: dict, detail_rects: dict) -> str:
    shots = assets["shots"]
    details = assets["details"]

    def shot_img(key, max_w):
        return data_url_for(BUILD / shots[key]["file"], max_w)

    collect_url = shot_img("collect", 3000)
    launcher_url = shot_img("launcher", 2400)
    edit_url = shot_img("edit", 1900)
    train_url = shot_img("train", 1900)
    preview_url = shot_img("preview", 1900)

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<title>AIoScout — Physical AI Workbench · Design System Poster</title>
<style>
{font_face_css()}
:root{{
  --canvas:#ffffff; --canvas-soft:#f9f9f8; --panel:#f3f4f3;
  --primary:#000000; --ink:#1a1c1c; --body:#525252; --mute:#5e5e5e;
  --hairline:#d8d9d8; --rule:#000000;
  --accent-orange:#f58331; --accent-purple:#5b5ba8;
  --success:#1f8a4c; --danger:#b3261e;
  --sans:'Inter', Arial, sans-serif;
  --mono:'JetBrains Mono','SFMono-Regular',Consolas,monospace;
}}
*{{box-sizing:border-box;border-radius:0!important;}}
html,body{{margin:0;padding:0;background:var(--canvas-soft);}}
body{{
  width:{POSTER_CSS_W}px;height:{POSTER_CSS_H}px;overflow:hidden;
  font-family:var(--sans);color:var(--ink);
  -webkit-font-smoothing:antialiased;text-rendering:geometricPrecision;
}}
.page{{
  width:{POSTER_CSS_W}px;height:{POSTER_CSS_H}px;background:var(--canvas-soft);
  padding:28px 34px 24px;display:flex;flex-direction:column;gap:10px;
}}
.page > *{{flex:0 0 auto;

}}
.mono{{font-family:var(--mono);font-variant-numeric:tabular-nums;}}
.eyebrow{{font-family:var(--mono);font-size:9.5px;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:var(--mute);}}
.eyebrow.o{{color:var(--accent-orange);}}
.eyebrow.p{{color:var(--accent-purple);}}
.rule{{height:1px;background:var(--rule);}}
.hairline{{height:1px;background:var(--hairline);}}

/* ---------- masthead ---------- */
.toprail{{display:flex;justify-content:space-between;align-items:baseline;font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--mute);}}
.mast{{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;}}
.brand{{display:flex;gap:16px;align-items:flex-start;min-width:0;}}
.mark{{width:48px;height:48px;color:var(--accent-orange);flex:0 0 48px;margin-top:2px;}}
.brand-copy{{min-width:0;}}
.brand-copy h1{{margin:0;font-size:40px;line-height:.94;font-weight:800;letter-spacing:-.035em;white-space:nowrap;}}
.brand-copy .sub{{margin:10px 0 0;font-size:13px;line-height:1.35;color:var(--body);max-width:610px;font-weight:600;letter-spacing:-.005em;}}
.brand-copy .sub em{{font-style:normal;color:var(--ink);}}
.meta{{width:392px;flex:0 0 392px;border:1px solid var(--rule);background:var(--canvas);}}
.meta .mrow{{display:grid;grid-template-columns:112px 1fr;border-top:1px solid var(--hairline);}}
.meta .mrow:first-child{{border-top:0;}}
.meta .mrow span{{padding:6px 8px;font-family:var(--mono);font-size:8.8px;line-height:1.25;}}
.meta .mrow span:first-child{{color:var(--mute);border-right:1px solid var(--hairline);letter-spacing:.04em;text-transform:uppercase;}}
.meta .mrow span:last-child{{font-weight:600;color:var(--ink);}}
.meta .mrow span b{{color:var(--accent-orange);font-weight:700;}}

/* ---------- section head ---------- */
.sechead{{display:flex;align-items:baseline;justify-content:space-between;gap:16px;}}
.sechead h2{{margin:0;font-size:12px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;}}
.sechead .secmeta{{font-family:var(--mono);font-size:9px;letter-spacing:.1em;color:var(--mute);text-transform:uppercase;}}

/* ---------- guidelines ---------- */
.guide{{display:grid;grid-template-columns:1fr 1.15fr 1.02fr 1.06fr;gap:0;border:1px solid var(--rule);background:var(--canvas);height:334px;}}
.gcol{{padding:11px 13px 10px;border-left:1px solid var(--rule);display:flex;flex-direction:column;min-width:0;overflow:hidden;}}
.gcol:first-child{{border-left:0;}}
.gcol .idx{{font-family:var(--mono);font-size:8.8px;letter-spacing:.14em;color:var(--accent-purple);font-weight:700;text-transform:uppercase;}}
.gcol h3{{margin:5px 0 8px;font-size:15px;font-weight:800;letter-spacing:-.02em;line-height:1.1;}}
.gcol p{{margin:0 0 6px;font-size:9.4px;line-height:1.42;color:var(--body);}}
.gcol p b{{color:var(--ink);font-weight:700;}}
.code{{font-family:var(--mono);font-size:8.8px;line-height:1.35;background:var(--panel);border:1px solid var(--hairline);padding:5px 6px;margin:2px 0 7px;color:var(--ink);letter-spacing:-.01em;}}
.tiers{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--rule);margin:2px 0 7px;}}
.tiers span{{height:24px;border-left:1px solid var(--rule);font-family:var(--mono);font-size:7.6px;display:flex;align-items:flex-end;padding:2px 3px;color:var(--mute);}}
.tiers span:first-child{{border-left:0;}}
.stack{{display:flex;gap:4px;height:40px;border:1px solid var(--rule);margin:2px 0 7px;align-items:stretch;}}
.stack div{{display:grid;place-items:center;font-family:var(--mono);font-size:7.5px;color:var(--mute);border-left:1px solid var(--hairline);}}
.stack div:first-child{{border-left:0;}}

.swgrid{{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--rule);}}
.sw{{display:grid;grid-template-columns:26px 1fr;gap:6px;align-items:center;padding:4px 0;border-bottom:1px solid var(--hairline);min-width:0;}}
.sw:nth-child(odd){{padding-right:8px;border-right:1px solid var(--hairline);}}
.sw:nth-child(even){{padding-left:8px;}}
.sw-chip{{display:block;width:26px;height:20px;border:1px solid var(--hairline);}}
.sw-meta{{display:flex;flex-direction:column;min-width:0;line-height:1.12;}}
.sw-meta b{{font-family:var(--mono);font-size:8.4px;letter-spacing:-.02em;}}
.sw-meta i{{font-family:var(--mono);font-size:7.2px;font-style:normal;color:var(--mute);}}
.sw-meta em{{display:none;}}
.roles{{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-top:auto;padding-top:5px;border-top:1px solid var(--hairline);}}
.roles span{{display:flex;align-items:center;gap:4px;font-family:var(--mono);font-size:7.2px;letter-spacing:.02em;color:var(--body);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.roles i{{width:8px;height:8px;flex:0 0 8px;border:1px solid var(--hairline);}}

.type-spec{{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:1px solid var(--hairline);padding-bottom:4px;margin-bottom:6px;}}
.type-spec .aa{{font-size:56px;font-weight:800;letter-spacing:-.05em;line-height:.8;}}
.type-spec .tag{{font-family:var(--mono);font-size:8px;line-height:1.3;color:var(--mute);text-align:right;}}
.type-row{{display:flex;align-items:baseline;justify-content:space-between;gap:8px;border-bottom:1px solid var(--hairline);padding:4px 0;}}
.type-row span:first-child{{flex:0 0 auto;}}
.type-row span:last-child{{font-family:var(--mono);font-size:7.6px;color:var(--mute);text-align:right;}}
.eyebrow-demo{{font-size:9px;letter-spacing:.17em;font-weight:700;text-transform:uppercase;}}
.mono-demo{{font-family:var(--mono);font-size:13px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.02em;}}
.floor{{margin-top:auto;display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:8.6px;color:var(--ink);border:1px solid var(--rule);padding:3px 6px;align-self:flex-start;letter-spacing:.04em;}}
.floor b{{background:var(--accent-orange);color:#fff;padding:1px 4px;font-size:11px;letter-spacing:.02em;}}

.stages{{display:flex;flex-direction:column;border:1px solid var(--rule);margin:2px 0 7px;}}
.stage-row{{display:grid;grid-template-columns:24px 1fr auto;align-items:center;gap:6px;border-top:1px solid var(--hairline);padding:4px 6px;}}
.stage-row:first-child{{border-top:0;}}
.stage-row b{{font-family:var(--mono);font-size:9px;color:var(--accent-orange);}}
.stage-row span{{font-size:9.6px;font-weight:700;letter-spacing:.02em;}}
.stage-row em{{font-family:var(--mono);font-size:7.4px;font-style:normal;color:var(--mute);text-transform:uppercase;letter-spacing:.05em;}}
.parity{{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid var(--rule);margin-top:auto;}}
.parity div{{padding:4px 6px;font-family:var(--mono);font-size:7.4px;line-height:1.3;color:var(--mute);border-left:1px solid var(--hairline);}}
.parity div:first-child{{border-left:0;}}
.parity b{{display:block;color:var(--ink);font-size:8.2px;margin-bottom:1px;}}

/* ---------- showcase ---------- */
.show-head{{display:flex;justify-content:space-between;align-items:baseline;border-top:1px solid var(--rule);padding-top:7px;}}
.show-head h2{{margin:0;font-size:16px;font-weight:800;letter-spacing:-.02em;}}
.show-head .r{{font-family:var(--mono);font-size:8.6px;letter-spacing:.1em;color:var(--mute);text-transform:uppercase;}}
.hero{{display:grid;grid-template-columns:660px 1fr;gap:14px;margin-top:4px;height:478px;}}
.shot{{position:relative;background:var(--canvas);border:1px solid var(--rule);overflow:hidden;}}
.shot img{{display:block;width:100%;height:100%;object-fit:cover;}}
.shot .frame-tag{{position:absolute;left:0;top:0;z-index:3;background:var(--primary);color:#fff;font-family:var(--mono);font-size:8px;font-weight:700;letter-spacing:.1em;padding:3px 6px;}}
.shot .frame-tag em{{font-style:normal;color:var(--accent-orange);}}
.hero-left{{display:flex;flex-direction:column;min-width:0;}}
.hero-left .shot{{height:411px;}}
.hero-right{{display:flex;flex-direction:column;gap:10px;min-width:0;}}
.hero-right .shot{{height:285px;}}
.cap{{display:flex;flex-direction:column;gap:4px;padding:5px 8px 6px;border:1px solid var(--rule);border-top:0;background:var(--canvas);}}
.callouts{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;padding-bottom:4px;border-bottom:1px solid var(--hairline);}}
.callouts span{{font-family:var(--mono);font-size:7.8px;font-weight:700;letter-spacing:.03em;line-height:1.2;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.callouts em{{font-style:normal;color:var(--accent-orange);margin-right:2px;}}

.cap p{{margin:0;font-size:8.4px;line-height:1.3;color:var(--body);}}
.cap p b{{color:var(--ink);font-weight:700;}}
.hero-right .note{{border:1px solid var(--rule);background:var(--panel);padding:6px 8px;display:flex;flex-direction:column;gap:3px;margin-top:auto;}}
.hero-right .note b{{font-family:var(--mono);font-size:8.4px;letter-spacing:.05em;}}
.hero-right .note span{{font-size:8.2px;line-height:1.3;color:var(--body);}}

/* leader callouts */
.leaders{{position:absolute;inset:0;width:100%;height:100%;z-index:2;pointer-events:none;}}
.leaders{{position:absolute;inset:0;width:100%;height:100%;z-index:2;pointer-events:none;}}
.leaders path{{fill:none;stroke:#000;stroke-width:1;vector-effect:non-scaling-stroke;opacity:.36;}}
.dot{{position:absolute;z-index:3;width:11px;height:11px;display:grid;place-items:center;background:var(--accent-orange);border:1px solid var(--primary);font-family:var(--mono);font-size:7.4px;font-weight:700;color:#000;line-height:1;transform:translate(-50%,-50%);}}

.workgrid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:10px;}}
.workcard{{border:1px solid var(--rule);background:var(--canvas);display:flex;flex-direction:column;}}
.workcard .shot{{border:0;border-bottom:1px solid var(--rule);height:229px;}}
.workcard .body{{padding:6px 8px 6px;display:flex;gap:7px;align-items:flex-start;}}
.workcard .idx{{font-family:var(--mono);font-size:15px;font-weight:700;color:var(--accent-orange);line-height:.9;}}
.workcard h3{{margin:0 0 2px;font-size:10px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;}}
.workcard p{{margin:0;font-size:8.2px;line-height:1.34;color:var(--body);}}
.workcard p b{{color:var(--ink);}}

.detailband{{display:grid;grid-template-columns:1fr 1.95fr;gap:12px;margin-top:10px;height:155px;}}
.detail{{border:1px solid var(--rule);background:var(--canvas);display:flex;flex-direction:row;min-width:0;overflow:hidden;}}
.detail .dshot{{flex:0 0 38%;height:100%;overflow:hidden;border-right:1px solid var(--rule);background:var(--canvas);display:flex;align-items:center;justify-content:center;min-height:0;padding:4px 4px 10px;}}
.detail.faders .dshot{{flex:0 0 62%;}}
.detail .dshot img{{display:block;width:auto;height:auto;max-width:100%;max-height:100%;object-fit:contain;}}
.detail .dnote{{padding:9px 11px;display:flex;flex-direction:column;gap:4px;min-width:0;}}
.detail .dnote b{{font-family:var(--mono);font-size:8.2px;letter-spacing:.06em;text-transform:uppercase;}}
.detail .dnote span{{font-size:8.4px;line-height:1.34;color:var(--body);}}
.detail .dnote em{{font-style:normal;color:var(--accent-orange);font-family:var(--mono);font-size:7.6px;letter-spacing:.05em;text-transform:uppercase;}}
.detail .dnote .sp{{margin-top:auto;font-family:var(--mono);font-size:7.2px;letter-spacing:.04em;color:var(--mute);text-transform:uppercase;}}

/* ---------- pipeline footer ---------- */
.pipe{{border:1px solid var(--rule);background:var(--canvas);padding:8px 10px;display:flex;flex-direction:column;gap:6px;}}
.pipe-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:0;}}
.pstep{{position:relative;border:1px solid var(--hairline);padding:6px 8px 6px;margin-right:14px;}}
.pstep:last-child{{margin-right:0;}}
.pstep:after{{content:'';position:absolute;right:-9px;top:50%;width:5px;height:5px;border-top:1px solid var(--primary);border-right:1px solid var(--primary);transform:translateY(-50%) rotate(45deg);}}
.pstep:last-child:after{{display:none;}}
.pstep b{{display:block;font-family:var(--mono);font-size:7.4px;color:var(--accent-orange);letter-spacing:.08em;}}
.pstep span{{display:block;font-size:8.7px;font-weight:700;line-height:1.25;margin-top:2px;}}
.pstep em{{display:block;font-family:var(--mono);font-size:7.2px;font-style:normal;color:var(--mute);margin-top:2px;line-height:1.3;}}
.colophon{{display:flex;justify-content:space-between;gap:12px;font-family:var(--mono);font-size:7.2px;letter-spacing:.05em;color:var(--mute);text-transform:uppercase;}}
</style>
</head>
<body>
<div class="page">

  <!-- ================= ZONE 1 — MASTHEAD ================= -->
  <div class="toprail">
    <span>AIoScout / Design System &amp; Interface Specimen</span>
    <span>Course submission · 2026.11 · Edition 01</span>
  </div>
  <div class="rule"></div>
  <div class="mast">
    <div class="brand">
      {LOGO_SVG}
      <div class="brand-copy">
        <div class="eyebrow p">Physical AI Workbench · zero-code INT8</div>
        <h1>AIoScout — Physical AI Workbench</h1>
        <p class="sub">Zero-Code INT8 Vision, OCR &amp; License-Plate Model Training System for <em>ESP32-P4</em> — capture, deterministic preprocessing, quantization-aware training, OOD-gated preview, and Arduino C++ export in one classroom instrument.</p>
      </div>
    </div>
    <div class="meta">
      <div class="mrow"><span>Target MCU</span><span>ESP32-P4 · 360 MHz · <b>RISC-V</b></span></div>
      <div class="mrow"><span>Quantization</span><span>Full INT8 FlatBuffer · ESP32-P4</span></div>
      <div class="mrow"><span>Pipelines</span><span>Image · 28×28 OCR · 3×28×28 Sign</span></div>
      <div class="mrow"><span>Aesthetic</span><span>Editorial-Tech Paper Instrument</span></div>
      <div class="mrow"><span>Capture</span><span>3× lossless · light theme · real DOM</span></div>
    </div>
  </div>

  <!-- ================= ZONE 2 — DESIGN GUIDELINES ================= -->
  <div class="sechead">
    <h2>Design Guidelines</h2>
    <div class="secmeta">04 rules · locked 2026-11-17 · normative</div>
  </div>
  <div class="guide">

    <div class="gcol">
      <div class="idx">01 · Geometry &amp; surface grammar</div>
      <h3>Zero-Ornamentation Rule</h3>
      <div class="code">border-radius: 0 !important;<br>box-shadow: none; · gradients: none;</div>
      <p><b>0px radius everywhere.</b> Precision oscilloscope and Braun / Teenage Engineering hardware grammar — rectangles, hairline seams, no soft edges.</p>
      <p><b>Depth = rules, not shadows.</b> No blur, no gradients; 1px structural rules + three paper tiers.</p>
      <div class="tiers">
        <span style="background:#ffffff">#FFFFFF</span>
        <span style="background:#f9f9f8">#F9F9F8</span>
        <span style="background:#f3f4f3">#F3F4F3</span>
      </div>
      <div class="stack">
        <div style="flex:0 0 56px;background:#f9f9f8">56</div>
        <div style="flex:1;background:#fff">1fr · workspace</div>
        <div style="flex:0 0 68px;background:#f9f9f8">68</div>
      </div>
      <p style="margin-bottom:0"><b>Zero-scroll fullscreen instrument:</b> fixed 100vw × 100vh, 56px hardware bar · 1fr stage · 68px four-stage footer.</p>
    </div>

    <div class="gcol">
      <div class="idx">02 · Colour system &amp; semantic tokens</div>
      <h3>Paper, Ink &amp; Signal</h3>
      <div class="swgrid">
        {''.join(swatch_html(*s) for s in SWATCHES)}
      </div>
      <p style="margin-top:6px;margin-bottom:0"><b>Orange is action and focus.</b> <b>Purple is annotation.</b> Green / red are reserved for distribution verdicts only — never decoration.</p>
      <div class="roles">
        <span><i style="background:#f58331"></i>FOCUS · ROI · TELEMETRY</span>
        <span><i style="background:#5b5ba8"></i>EYEBROW · OCR · LOSS</span>
        <span><i style="background:#1f8a4c"></i>IN-DISTRIBUTION · READY</span>
        <span><i style="background:#b3261e"></i>OOD GATE · DESTRUCTIVE</span>
      </div>
    </div>

    <div class="gcol">
      <div class="idx">03 · Typography &amp; readability floor</div>
      <h3>Inter + JetBrains Mono</h3>
      <div class="type-spec">
        <span class="aa">Aa</span>
        <span class="tag">INTER · 400 / 600 / 700 / 800<br>display −0.035em</span>
      </div>
      <div class="type-row"><span class="eyebrow-demo o">STAGE 01 · COLLECT</span><span>700 · UPPER · .17em</span></div>
      <div class="type-row"><span style="font-size:9.3px;font-weight:600">Classroom-friendly INT8 training, local.</span><span>400 / 600 · 9–13px</span></div>
      <div class="type-row"><span class="mono-demo">96.7% · 0.041 · 1.18 MB</span><span>MONO · TABULAR</span></div>
      <div class="type-row"><span class="mono" style="font-size:9.2px">DARK 0 · LUM 100 · ENTR 0.85</span><span>FADER READOUT</span></div>
      <p style="margin-top:6px">Telemetry uses <b>tabular numerals</b> so counters, faders and probabilities never jitter horizontally as values change.</p>
      <div class="floor"><b>10PX</b> UI HARD MINIMUM TYPE FLOOR · BADGES · AXIS TICKS · MATRIX CELLS</div>
    </div>

    <div class="gcol">
      <div class="idx">04 · Interaction &amp; parity</div>
      <h3>Stage Segregation &amp; Parity</h3>
      <div class="stages">
        <div class="stage-row"><b>01</b><span>Collect</span><em>2 decks · 24 strip</em></div>
        <div class="stage-row"><b>02</b><span>Edit</span><em>ROI · Otsu</em></div>
        <div class="stage-row"><b>03</b><span>Train</span><em>INT8 telemetry</em></div>
        <div class="stage-row"><b>04</b><span>Preview / Export</span><em>OOD gate</em></div>
      </div>
      <p><b>Strict stage separation</b> — no cross-stage modal clutter. Within the workbench, the four-stage footer is the only global navigation.</p>
      <p><b>Bit-exact Python / C++ parity.</b> Desktop preview and exported ESP32-P4 firmware headers share one deterministic contract:</p>
      <div class="parity">
        <div><b>LUMINANCE</b>BT.601 30/59/11 · round half up</div>
        <div><b>WINDOW</b>Otsu + dark/lum gate</div>
        <div><b>QUANTIZE</b>gray − 128 · scale 1/255</div>
        <div><b>OOD</b>max-prob + entropy gate</div>
      </div>
    </div>
  </div>

  <!-- ================= ZONE 3 — SHOWCASE ================= -->
  <div class="show-head">
    <h2>Annotated interface showcase</h2>
    <div class="r">Real DOM captures · light / editorial paper · 3× device scale</div>
  </div>

  <div class="hero">
    <div class="hero-left">
      <div class="shot">
        <img src="{collect_url}" alt="AIoScout 01 Collect — two-deck class board, whitelisted plate strip, live stage, 24-sample strip">
        <svg class="leaders" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <path d="M24,22 V100"/><path d="M66.5,49.6 V100"/><path d="M10,12.5 V100"/>
        </svg>
        <span class="dot" style="left:24%;top:22%">1</span>
        <span class="dot" style="left:66.5%;top:49.6%">2</span>
        <span class="dot" style="left:10%;top:12.5%">3</span>
      </div>
      <div class="cap">
        <div class="callouts">
          <span><em>01</em> 2-deck class rows</span>
          <span><em>02</em> lock-free capture strip</span>
          <span><em>03</em> single-row whitelist</span>
        </div>
        <p><b>SHOT B · 01 COLLECT — Two-deck class board + single-row whitelist.</b> Nine slot-glyph classes derive from three plates; a 24-sample incremental strip sits under a lock-free capture bar, so teachers can audit purity without leaving the stage.</p>
      </div>
    </div>

    <div class="hero-right">
      <div class="shot">
        <img src="{launcher_url}" alt="AIoScout launcher — brand mark, INT8 architecture table, square triggers, recent projects">
        <svg class="leaders" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <path d="M35.7,52 V100"/><path d="M74,9 V100"/><path d="M64,36 V100"/>
        </svg>
        <span class="dot" style="left:35.7%;top:52%">1</span>
        <span class="dot" style="left:74%;top:9%">2</span>
        <span class="dot" style="left:64%;top:36%">3</span>
      </div>
      <div class="cap">
        <div class="callouts">
          <span><em>01</em> 5fr : 9fr golden split</span>
          <span><em>02</em> square triggers</span>
          <span><em>03</em> recent ledger</span>
        </div>
        <p><b>SHOT A · LAUNCHER — 5fr : 9fr golden split.</b> Brand + INT8 pipeline architecture on the left; NEW / OPEN / SETTINGS triggers and the recent-project ledger on the right.</p>
      </div>
      <div class="note">
        <b>ONE STAGE MODEL · ZERO MODE SWITCHES</b>
        <span>Image · OCR · SIGN_OCR share the same four-stage workbench. The project type only changes the tensor contract — never the interaction grammar.</span>
      </div>
    </div>
  </div>

  <div class="workgrid">
    <article class="workcard">
      <div class="shot">
        <img src="{edit_url}" alt="AIoScout 02 Edit — class TOC, original sample with 8-handle ROI, processed tensor, threshold inspector, Otsu readout">
      </div>
      <div class="body"><span class="idx">02</span><div><h3>Shot C · 02 Edit — purity workbench</h3><p>Three zones: class TOC · original + <b>8-handle ROI</b> + processed tensor · threshold, pixel &amp; Otsu inspector. Auto Search / Manual ROI are one segmented contract.</p></div></div>
    </article>

    <article class="workcard">
      <div class="shot">
        <img src="{train_url}" alt="AIoScout 03 Train — dataset audit, training parameters, KPI grid, confusion matrix, accuracy and loss curves">
      </div>
      <div class="body"><span class="idx">03</span><div><h3>Shot D · 03 Train — telemetry instrument</h3><p>Zero-scroll two-column run: dataset audit bars, parameters, four KPIs, per-class confusion matrix and accuracy / loss curves — no navigation required.</p></div></div>
    </article>

    <article class="workcard">
      <div class="shot">
        <img src="{preview_url}" alt="AIoScout 04 Preview — live inference with green ROI overlay, whitelist verdict, six faders, INT8 C++ export card">
      </div>
      <div class="body"><span class="idx">04</span><div><h3>Shot E · 04 Preview + export</h3><p>Green ROI and slot boxes over the live stage; verdict board underneath; <b>six vertical faders</b> for dark / lum / max-p / entropy / area; INT8 C++ export card.</p></div></div>
    </article>
  </div>

  <div class="detailband">
    <div class="detail roi">
      <div class="dshot"><img src="{details['roi']}" alt="Detail: 8-handle ROI on the original sample"></div>
      <div class="dnote">
        <b>DETAIL · 02 EDIT / 8-HANDLE ROI</b>
        <span>Auto Search or Manual ROI on the same <b>1px orange</b> frame: eight resize handles, live crop readout, and the suggested Otsu threshold beside the processed tensor.</span>
        <span class="sp">{details['roi_line']}</span>
      </div>
    </div>
    <div class="detail faders">
      <div class="dshot"><img src="{details['faders']}" alt="Detail: six vertical threshold and OOD faders"></div>
      <div class="dnote">
        <b>DETAIL · 04 PREVIEW / PRE- &amp; OOD FADER BANK</b>
        <span>Six vertical instrument faders share the same numeric fields as the exported header: <b>DARK · LUM · MAX-P · ENTR · AREA-MIN · AREA-MAX</b>. The orange thumb is the only moving accent; the black fill is the committed threshold.</span>
        <span class="sp">TABULAR READOUT · 0.40 MAX-P / 0.85 ENTROPY · P4-PARITY GATE</span>
      </div>
    </div>
  </div>

  <!-- ================= ZONE 4 — PIPELINE FOOTER ================= -->
  <div class="pipe">
    <div class="eyebrow p">End-to-end pipeline · capture to firmware</div>
    <div class="pipe-row">
      <div class="pstep"><b>01 · INPUT</b><span>Live UVC / ESP32-P4 Serial</span><em>AA 55 AA frame stream</em></div>
      <div class="pstep"><b>02 · DETERMINISTIC</b><span>ROI &amp; threshold cache</span><em>BT.601 · Otsu · int8</em></div>
      <div class="pstep"><b>03 · LEARN</b><span>INT8 quantization-aware training</span><em>confusion · calibration</em></div>
      <div class="pstep"><b>04 · GATE</b><span>Real-time OOD gatekeeper</span><em>max-prob · entropy</em></div>
      <div class="pstep"><b>05 · SHIP</b><span>Arduino C++ FlatBuffer bundle</span><em>.tflite + model_settings.h</em></div>
    </div>
  </div>
  <div class="colophon">
    <span>AIoScout · Physical AI Workbench · DESIGN SYSTEM SPECIMEN</span>
    <span>Inter + JetBrains Mono · SIL OFL 1.1 · captures unretouched at 3× · 3600 × 5100 px</span>
  </div>

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# poster render + verification
# ---------------------------------------------------------------------------

def render_poster(html_path: Path, png_path: Path, pdf_path: Path, preview_path: Path):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-color-profile=srgb", "--font-render-hinting=none"])
        context = browser.new_context(
            viewport={"width": POSTER_CSS_W, "height": POSTER_CSS_H},
            device_scale_factor=DSF,
            color_scheme="light",
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(html_path.as_uri())
        page.wait_for_load_state("load")
        page.evaluate("() => document.fonts.ready")
        page.wait_for_timeout(700)

        audit = page.evaluate(
            """
            () => {
              const imgs = Array.from(document.images).map((img) => ({
                src: (img.currentSrc || img.src).slice(0, 42),
                ok: img.complete && img.naturalWidth > 0,
                w: img.naturalWidth, h: img.naturalHeight
              }));
              const overflow = [];
              document.querySelectorAll('.page *').forEach((el) => {
                if (el.closest('svg') || el.classList.contains('pstep')) return;
                if (el.scrollWidth > el.clientWidth + 2 && el.clientWidth > 0) {
                  overflow.push({ tag: el.tagName, cls: String(el.className).slice(0, 40),
                                  sw: el.scrollWidth, cw: el.clientWidth });
                }
              });
              const vclip = [];
              document.querySelectorAll('.gcol,.cap,.dnote,.workcard .body,.pstep,.brand-copy,.meta,.shot,.detail,.floor,.type-row,.colophon,.page').forEach((el) => {
                if (el.scrollHeight > el.clientHeight + 2 && el.clientHeight > 0) {
                  vclip.push({ cls: String(el.className).slice(0, 40), sh: el.scrollHeight, ch: el.clientHeight });
                }
              });
              const oop = [];
              document.querySelectorAll('.page *').forEach((el) => {
                if (el.closest('svg') || !el.textContent || !el.textContent.trim()) return;
                if (el.children.length) return;
                const r = el.getBoundingClientRect();
                if (r.width < 0.5 || r.height < 0.5) return;
                if (r.left < -0.5 || r.top < -0.5 || r.right > 1200.5 || r.bottom > 1700.5) {
                  oop.push({ cls: String(el.className).slice(0, 40), text: el.textContent.trim().slice(0, 24),
                             box: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)] });
                }
              });
              const body = document.body.getBoundingClientRect();
              const pageBox = document.querySelector('.page').getBoundingClientRect();
              return {
                imgs,
                overflow: overflow.slice(0, 12),
                vclip: vclip.slice(0, 12),
                oop: oop.slice(0, 12),
                body: { w: body.width, h: body.height, sh: document.documentElement.scrollHeight },
                pageBox: { w: pageBox.width, h: pageBox.height },
                fonts: {
                  inter800: document.fonts.check("800 12px Inter"),
                  inter400: document.fonts.check("400 10px Inter"),
                  mono600: document.fonts.check("600 10px 'JetBrains Mono'")
                }
              };
            }
            """
        )
        if errors:
            raise RuntimeError("Poster raised JS errors:\n  " + "\n  ".join(errors))
        broken = [i for i in audit["imgs"] if not i["ok"]]
        if broken:
            raise RuntimeError(f"Broken poster images: {broken}")
        if audit["overflow"]:
            raise RuntimeError(f"Overflowing poster elements: {audit['overflow']}")
        if audit["vclip"]:
            raise RuntimeError(f"Clipped poster elements: {audit['vclip']}")
        if audit["oop"]:
            raise RuntimeError(f"Text outside the poster page: {audit['oop']}")
        if (abs(audit["pageBox"]["w"] - POSTER_CSS_W) > 0.6
                or abs(audit["pageBox"]["h"] - POSTER_CSS_H) > 0.6
                or audit["body"]["sh"] > POSTER_CSS_H + 1):
            raise RuntimeError(f"Poster page size mismatch: {audit}")
        if not all(audit["fonts"].values()):
            raise RuntimeError(f"Poster fonts failed to load: {audit['fonts']}")

        page.screenshot(path=str(png_path), full_page=False, animations="disabled")
        page.pdf(
            path=str(pdf_path),
            width="420mm",
            height="594mm",
            scale=1.3203,
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        # small inspection copy
        shot = Image.open(png_path)
        shot.resize((1200, 1700), Image.LANCZOS).save(preview_path)
        context.close()
        browser.close()
        return audit


def verify_outputs(png_path: Path, pdf_path: Path):
    im = Image.open(png_path)
    if im.size != (3600, 5100):
        raise RuntimeError(f"PNG is {im.size}, expected (3600, 5100)")
    pdf = pdf_path.read_bytes()
    boxes = re.findall(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    if len(boxes) != 1:
        raise RuntimeError(f"PDF has {len(boxes)} MediaBox entries, expected exactly 1")
    x0, y0, w, h = (float(v) for v in boxes[0])
    if abs(x0) > 0.01 or abs(y0) > 0.01:
        raise RuntimeError(f"PDF MediaBox origin {x0},{y0}, expected 0,0")
    if abs(w - 1190.55) > 1.0 or abs(h - 1683.78) > 1.0:
        raise RuntimeError(f"PDF MediaBox {w}x{h} pt, expected A2 1190.55x1683.78")
    if abs((w / h) - (2 ** -0.5)) > 0.001:
        raise RuntimeError(f"PDF ratio {w/h:.4f} is not A-series (1:sqrt2)")
    return {"png": im.size, "pdf_pt": (w, h)}


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    ensure_fonts()
    shell_path = make_capture_shell()
    print(f"[1/4] capture shell  {shell_path.relative_to(REPO)}")
    shots = capture_views(shell_path)
    print(f"[2/4] captured 5 views at {CAPTURE_W}x{CAPTURE_H} @ {DSF}x")

    rects = json.loads((BUILD / "detail_rects.json").read_text())
    r = rects["rects"]
    # detail crops (real DOM regions from the same 3x screenshots)
    edit_shot = BUILD / "shot_c_edit.png"
    preview_shot = BUILD / "shot_e_preview.png"
    roi_rect = {
        "x": r["editRoi"]["x"] - 34,
        "y": r["editRoi"]["y"] - 34,
        "w": r["editRoi"]["w"] + 68,
        "h": r["editRoi"]["h"] + 68,
    }
    readouts = rects.get("readouts") or {}
    crop_txt = readouts.get("crop") or "—"
    otsu_txt = readouts.get("otsu") or "—"
    details = {
        "roi": crop_bytes(edit_shot, roi_rect, 900),
        "faders": crop_bytes(preview_shot, {**r["faderBank"], "h": r["faderBank"]["h"] + 6}, 1800, pad=8),
        "roi_line": f"AUTO SEARCH / MANUAL ROI · CROP {crop_txt} · OTSU {otsu_txt}",
    }
    assets = {
        "shots": {
            "launcher": {"file": "shot_a_launcher.png"},
            "collect": {"file": "shot_b_collect.png"},
            "edit": {"file": "shot_c_edit.png"},
            "train": {"file": "shot_d_train.png"},
            "preview": {"file": "shot_e_preview.png"},
        },
        "details": details,
    }
    html = poster_html(assets, rects)
    html_build = BUILD / "AIoScout_Design_Poster.html"
    png_build = BUILD / "AIoScout_Design_Poster.png"
    pdf_build = BUILD / "AIoScout_Design_Poster.pdf"
    preview_path = BUILD / "poster_preview.png"
    html_build.write_text(html, encoding="utf-8")
    print(f"[3/4] poster html     {html_build.relative_to(REPO)}  ({html_build.stat().st_size/1e6:.1f} MB)")

    audit = render_poster(html_build, png_build, pdf_build, preview_path)
    dims = verify_outputs(png_build, pdf_build)

    # Publish atomically: verify the whole set in BUILD first, then replace the
    # three deliverables together.  A failure above leaves the prior set intact.
    html_path = DOCS / "AIoScout_Design_Poster.html"
    png_path = DOCS / "AIoScout_Design_Poster.png"
    pdf_path = DOCS / "AIoScout_Design_Poster.pdf"
    for src, dst in ((html_build, html_path), (png_build, png_path), (pdf_build, pdf_path)):
        os.replace(src, dst)
    print(f"[4/4] rendered        {png_path.name} {dims['png']} · {pdf_path.name} {dims['pdf_pt']} pt")
    print(f"      images={len(audit['imgs'])} broken=0 overflow=0 vclip=0 oop=0 fonts=ok")
    print(f"      inspection copy {preview_path}")


if __name__ == "__main__":
    main()
