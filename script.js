/**
 * AirDraw AI Infinity — script.js
 * Complete production-ready hand-tracking air drawing application.
 *
 * Architecture:
 *  - AppConfig       : constants and configuration
 *  - Utils           : helpers (smoothing, geometry, math)
 *  - CanvasManager   : manages 3-layer canvas system
 *  - ColorPalette    : preset colors + custom picker
 *  - BrushEngine     : renders strokes with multiple brush types
 *  - StrokeBuffer    : accumulates points for one stroke
 *  - HistoryManager  : undo / redo stack
 *  - ShapeRecognizer : detects and renders geometric shapes
 *  - GestureDetector : classifies hand landmarks into gestures
 *  - HandTracker     : wraps MediaPipe Hands
 *  - CameraManager   : handles camera permissions and streams
 *  - UIController    : binds DOM elements, updates HUD
 *  - AirDrawApp      : top-level orchestrator
 */

'use strict';

/* ═══════════════════════════════════════════════════════
   1. APP CONFIG
═══════════════════════════════════════════════════════ */
const AppConfig = Object.freeze({
  // MediaPipe settings
  MEDIAPIPE_MODEL_COMPLEXITY: 0,          // 0=lite (fast), 1=full
  MEDIAPIPE_MIN_DETECTION_CONFIDENCE: 0.7,
  MEDIAPIPE_MIN_TRACKING_CONFIDENCE: 0.6,
  MEDIAPIPE_MAX_NUM_HANDS: 1,

  // Smoothing
  SMOOTHING_FACTOR: 0.55,               // 0=no smooth, 1=max smooth (Exponential Moving Average)
  SMOOTHING_HISTORY: 5,                  // points for velocity-weighted avg

  // Stroke rendering
  MIN_STROKE_DISTANCE: 2,               // px — skip point if closer than this
  STROKE_JOIN: 'round',
  STROKE_CAP: 'round',
  DEFAULT_COLOR: '#a78bfa',
  DEFAULT_BRUSH_SIZE: 8,
  DEFAULT_OPACITY: 1.0,
  DEFAULT_TOOL: 'pencil',

  // Gesture thresholds
  PINCH_THRESHOLD: 0.07,               // normalized distance < this = pinch
  FIST_THRESHOLD: 0.12,                // all fingertips folded
  OPEN_PALM_THRESHOLD: 0.18,           // all fingers extended

  // Drawing state machine
  DRAW_COOLDOWN_MS: 80,                // prevent state flicker

  // Shape recognition
  SHAPE_MIN_POINTS: 12,
  SHAPE_CLOSE_THRESHOLD: 0.15,         // ratio of bbox diagonal for closed shape
  SHAPE_STRAIGHTNESS_THRESHOLD: 0.93,  // R² for straight line
  SHAPE_CIRCLE_THRESHOLD: 0.82,        // circularity score
  SHAPE_DETECTION_DELAY_MS: 320,       // wait after stroke ends before analyzing

  // History
  MAX_HISTORY: 40,

  // FPS
  FPS_UPDATE_INTERVAL: 500,            // ms between FPS display updates

  // Notifications
  TOAST_DURATION_MS: 2200,
  SHAPE_TOAST_DURATION_MS: 1600,

  // Camera
  IDEAL_WIDTH: 1280,
  IDEAL_HEIGHT: 720,
  IDEAL_FPS: 60,

  // Preset colors (25)
  PALETTE_COLORS: [
    '#ffffff', '#e2e8f0', '#94a3b8', '#475569', '#1e293b',
    '#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399',
    '#22d3ee', '#38bdf8', '#818cf8', '#a78bfa', '#f472b6',
    '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6',
    '#0ea5e9', '#6366f1', '#8b5cf6', '#ec4899', '#06b6d4',
  ],

  // Brush opacity for each type (relative multipliers)
  BRUSH_CONFIGS: {
    pencil: { globalAlpha: 1.0,  lineWidthMultiplier: 1.0,  glow: false, composite: 'source-over' },
    marker: { globalAlpha: 0.75, lineWidthMultiplier: 2.2,  glow: false, composite: 'source-over' },
    neon:   { globalAlpha: 0.95, lineWidthMultiplier: 1.0,  glow: true,  composite: 'source-over' },
    eraser: { globalAlpha: 1.0,  lineWidthMultiplier: 3.0,  glow: false, composite: 'destination-out' },
  },
});

/* ═══════════════════════════════════════════════════════
   2. UTILS
═══════════════════════════════════════════════════════ */
const Utils = {
  /**
   * Exponential moving average for a value.
   */
  ema(prev, next, alpha) {
    return prev + alpha * (next - prev);
  },

  /**
   * Euclidean distance between two points.
   */
  dist(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  },

  /**
   * Midpoint between two points.
   */
  midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  },

  /**
   * Clamp a value between min and max.
   */
  clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  },

  /**
   * Convert hex color to rgba string.
   */
  hexToRgba(hex, alpha = 1) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  },

  /**
   * Linear interpolation between two numbers.
   */
  lerp(a, b, t) {
    return a + (b - a) * t;
  },

  /**
   * Debounce function.
   */
  debounce(fn, ms) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  },

  /**
   * Throttle function.
   */
  throttle(fn, ms) {
    let last = 0;
    return (...args) => {
      const now = performance.now();
      if (now - last >= ms) {
        last = now;
        fn.apply(this, args);
      }
    };
  },

  /**
   * Compute bounding box of a set of points.
   */
  boundingBox(points) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    return { minX, minY, maxX, maxY, w: maxX - minX, h: maxY - minY };
  },

  /**
   * Normalize points to 0-1 range based on their bounding box.
   */
  normalizePoints(points) {
    const bb = Utils.boundingBox(points);
    const span = Math.max(bb.w, bb.h, 1);
    return points.map(p => ({
      x: (p.x - bb.minX) / span,
      y: (p.y - bb.minY) / span,
    }));
  },

  /**
   * Calculate approximate arc length of a polyline.
   */
  polylineLength(points) {
    let total = 0;
    for (let i = 1; i < points.length; i++) {
      total += Utils.dist(points[i - 1], points[i]);
    }
    return total;
  },

  /**
   * Simplify a polyline using Ramer-Douglas-Peucker algorithm.
   */
  rdpSimplify(points, epsilon) {
    if (points.length < 3) return points.slice();
    let maxDist = 0;
    let maxIdx  = 0;
    const first = points[0];
    const last  = points[points.length - 1];
    const dx = last.x - first.x;
    const dy = last.y - first.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    for (let i = 1; i < points.length - 1; i++) {
      // Perpendicular distance from point to line
      const d = Math.abs(dy * points[i].x - dx * points[i].y + last.x * first.y - last.y * first.x) / len;
      if (d > maxDist) { maxDist = d; maxIdx = i; }
    }
    if (maxDist > epsilon) {
      const left  = Utils.rdpSimplify(points.slice(0, maxIdx + 1), epsilon);
      const right = Utils.rdpSimplify(points.slice(maxIdx), epsilon);
      return left.slice(0, -1).concat(right);
    }
    return [first, last];
  },

  /**
   * Compute centroid of a set of points.
   */
  centroid(points) {
    let sx = 0, sy = 0;
    for (const p of points) { sx += p.x; sy += p.y; }
    return { x: sx / points.length, y: sy / points.length };
  },
};

/* ═══════════════════════════════════════════════════════
   3. CANVAS MANAGER
═══════════════════════════════════════════════════════ */
class CanvasManager {
  constructor() {
    this.cameraCanvas  = document.getElementById('camera-canvas');
    this.drawingCanvas = document.getElementById('drawing-canvas');
    this.overlayCanvas = document.getElementById('overlay-canvas');

    this.cameraCtx  = this.cameraCanvas.getContext('2d');
    this.drawingCtx = this.drawingCanvas.getContext('2d');
    this.overlayCtx = this.overlayCanvas.getContext('2d');

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(document.getElementById('canvas-container'));

    this._onResize();
  }

  get width()  { return this.drawingCanvas.width; }
  get height() { return this.drawingCanvas.height; }

  _onResize() {
    const container = document.getElementById('canvas-container');
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;

    for (const canvas of [this.cameraCanvas, this.drawingCanvas, this.overlayCanvas]) {
      canvas.width  = w;
      canvas.height = h;
    }
    // Trigger event so app can redraw preserved content
    window.dispatchEvent(new CustomEvent('canvasresize', { detail: { w, h } }));
  }

  /**
   * Draw a mirrored camera frame onto the camera canvas.
   */
  drawCameraFrame(videoEl) {
    if (!videoEl || videoEl.readyState < 2) return;
    const ctx = this.cameraCtx;
    const w = this.width;
    const h = this.height;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, 0, 0, w, h);
    ctx.restore();
  }

  /**
   * Clear the overlay canvas.
   */
  clearOverlay() {
    this.overlayCtx.clearRect(0, 0, this.width, this.height);
  }

  /**
   * Clear the drawing canvas.
   */
  clearDrawing() {
    this.drawingCtx.clearRect(0, 0, this.width, this.height);
  }

  /**
   * Get drawing canvas image data for undo.
   */
  getDrawingSnapshot() {
    return this.drawingCtx.getImageData(0, 0, this.width, this.height);
  }

  /**
   * Restore drawing canvas from a snapshot.
   */
  restoreDrawingSnapshot(imageData) {
    this.drawingCtx.putImageData(imageData, 0, 0);
  }

  /**
   * Composite drawing + camera to a flat PNG for export.
   */
  exportPNG() {
    const offscreen = document.createElement('canvas');
    offscreen.width  = this.width;
    offscreen.height = this.height;
    const ctx = offscreen.getContext('2d');
    ctx.drawImage(this.cameraCanvas, 0, 0);
    ctx.drawImage(this.drawingCanvas, 0, 0);
    return offscreen.toDataURL('image/png');
  }
}

/* ═══════════════════════════════════════════════════════
   4. HISTORY MANAGER
═══════════════════════════════════════════════════════ */
class HistoryManager {
  constructor(canvasManager) {
    this._cm    = canvasManager;
    this._stack = [];
    this._index = -1;
  }

  /**
   * Push current drawing canvas state to history.
   */
  push() {
    // Remove any redo states above current index
    this._stack.splice(this._index + 1);
    const snap = this._cm.getDrawingSnapshot();
    this._stack.push(snap);
    // Cap size
    if (this._stack.length > AppConfig.MAX_HISTORY) {
      this._stack.shift();
    }
    this._index = this._stack.length - 1;
  }

  /**
   * Undo last drawing operation.
   */
  undo() {
    if (this._index <= 0) {
      // Revert to blank
      if (this._index === 0) {
        this._index = -1;
        this._cm.clearDrawing();
      }
      return false;
    }
    this._index--;
    this._cm.restoreDrawingSnapshot(this._stack[this._index]);
    return true;
  }

  /**
   * Redo last undone operation.
   */
  redo() {
    if (this._index >= this._stack.length - 1) return false;
    this._index++;
    this._cm.restoreDrawingSnapshot(this._stack[this._index]);
    return true;
  }

  get canUndo() { return this._index >= 0; }
  get canRedo() { return this._index < this._stack.length - 1; }

  get stackInfo() {
    return `${this._index + 1}/${this._stack.length}`;
  }
}

/* ═══════════════════════════════════════════════════════
   5. BRUSH ENGINE
═══════════════════════════════════════════════════════ */
class BrushEngine {
  constructor(canvasManager) {
    this._cm    = canvasManager;
    this.color  = AppConfig.DEFAULT_COLOR;
    this.size   = AppConfig.DEFAULT_BRUSH_SIZE;
    this.opacity = AppConfig.DEFAULT_OPACITY;
    this.tool   = AppConfig.DEFAULT_TOOL;
  }

  /**
   * Begin a new stroke path — sets context properties.
   */
  beginStroke(ctx) {
    const cfg = AppConfig.BRUSH_CONFIGS[this.tool] || AppConfig.BRUSH_CONFIGS.pencil;
    ctx.save();
    ctx.globalCompositeOperation = cfg.composite;
    ctx.globalAlpha  = Utils.clamp(this.opacity * cfg.globalAlpha, 0, 1);
    ctx.strokeStyle  = this.color;
    ctx.lineWidth    = this.size * cfg.lineWidthMultiplier;
    ctx.lineCap      = AppConfig.STROKE_CAP;
    ctx.lineJoin     = AppConfig.STROKE_JOIN;

    if (cfg.glow) {
      ctx.shadowColor = this.color;
      ctx.shadowBlur  = this.size * 3;
    }
  }

  /**
   * End stroke — restore context.
   */
  endStroke(ctx) {
    ctx.restore();
  }

  /**
   * Render a smooth stroke through an array of points using quadratic curves.
   * Uses midpoint technique to prevent gaps.
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<{x,y}>} points
   */
  renderStroke(ctx, points) {
    if (points.length < 2) {
      // Render a dot for single-point strokes
      if (points.length === 1) {
        this.beginStroke(ctx);
        const cfg = AppConfig.BRUSH_CONFIGS[this.tool] || AppConfig.BRUSH_CONFIGS.pencil;
        const r = (this.size * cfg.lineWidthMultiplier) / 2;
        ctx.beginPath();
        ctx.arc(points[0].x, points[0].y, Math.max(r, 1), 0, Math.PI * 2);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fill();
        this.endStroke(ctx);
      }
      return;
    }

    this.beginStroke(ctx);
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length - 1; i++) {
      const mid = Utils.midpoint(points[i], points[i + 1]);
      ctx.quadraticCurveTo(points[i].x, points[i].y, mid.x, mid.y);
    }

    // Last point
    const last = points[points.length - 1];
    ctx.lineTo(last.x, last.y);
    ctx.stroke();

    // Second neon pass for glow effect
    if (AppConfig.BRUSH_CONFIGS[this.tool]?.glow) {
      ctx.globalAlpha = 0.35;
      ctx.lineWidth   = this.size * 0.4;
      ctx.shadowBlur  = this.size * 6;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }

    this.endStroke(ctx);
  }

  /**
   * Render a single segment from lastPt to curPt — incremental stroke extension.
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<{x,y}>} recentPoints  last ~4 points for smooth curve
   */
  renderSegment(ctx, recentPoints) {
    if (recentPoints.length < 2) return;
    this.renderStroke(ctx, recentPoints);
  }
}

/* ═══════════════════════════════════════════════════════
   6. STROKE BUFFER
═══════════════════════════════════════════════════════ */
class StrokeBuffer {
  constructor() {
    this.points = [];
    this._smoothX = null;
    this._smoothY = null;
  }

  reset() {
    this.points  = [];
    this._smoothX = null;
    this._smoothY = null;
  }

  /**
   * Add a raw point, applying EMA smoothing.
   * Returns the smoothed point.
   * @param {number} rawX
   * @param {number} rawY
   * @returns {{x:number, y:number}|null}
   */
  addPoint(rawX, rawY) {
    if (this._smoothX === null) {
      this._smoothX = rawX;
      this._smoothY = rawY;
    } else {
      const alpha = 1 - AppConfig.SMOOTHING_FACTOR;
      this._smoothX = Utils.ema(this._smoothX, rawX, alpha);
      this._smoothY = Utils.ema(this._smoothY, rawY, alpha);
    }

    const smoothPt = { x: this._smoothX, y: this._smoothY };

    // Only add if moved enough
    if (this.points.length > 0) {
      const last = this.points[this.points.length - 1];
      if (Utils.dist(last, smoothPt) < AppConfig.MIN_STROKE_DISTANCE) return null;
    }

    this.points.push(smoothPt);
    return smoothPt;
  }

  /**
   * Get the last N points (for segment rendering).
   */
  tail(n = 4) {
    return this.points.slice(Math.max(0, this.points.length - n));
  }

  get length() { return this.points.length; }

  get isEmpty() { return this.points.length === 0; }
}

/* ═══════════════════════════════════════════════════════
   7. SHAPE RECOGNIZER
═══════════════════════════════════════════════════════ */
class ShapeRecognizer {
  /**
   * Attempt to classify a stroke into a known shape.
   * Returns { shape, confidence } or null.
   * @param {Array<{x,y}>} rawPoints
   */
  recognize(rawPoints) {
    if (rawPoints.length < AppConfig.SHAPE_MIN_POINTS) return null;

    // Simplify for analysis (not for rendering)
    const pts  = Utils.rdpSimplify(rawPoints, 4);
    const bb   = Utils.boundingBox(pts);

    // Require minimum size
    if (bb.w < 20 && bb.h < 20) return null;

    const isClosedShape = this._isClosed(rawPoints);
    const straightness  = this._straightness(rawPoints);

    // ── Straight Line / Arrow ───────────────────────────
    if (!isClosedShape && straightness > AppConfig.SHAPE_STRAIGHTNESS_THRESHOLD) {
      // Check if it could be an arrow (has a sharp direction change at end)
      if (this._hasArrowhead(rawPoints)) {
        return { shape: 'arrow', confidence: 0.88 };
      }
      return { shape: 'line', confidence: straightness };
    }

    if (!isClosedShape) return null;

    // ── Circle ──────────────────────────────────────────
    const circularity = this._circularity(rawPoints);
    if (circularity > AppConfig.SHAPE_CIRCLE_THRESHOLD) {
      return { shape: 'circle', confidence: circularity };
    }

    // ── Triangle ────────────────────────────────────────
    if (pts.length >= 3 && pts.length <= 7) {
      const triScore = this._triangleScore(rawPoints);
      if (triScore > 0.7) {
        return { shape: 'triangle', confidence: triScore };
      }
    }

    // ── Square / Rectangle ──────────────────────────────
    const rectScore = this._rectangleScore(rawPoints);
    if (rectScore > 0.68) {
      const aspect = bb.w / (bb.h || 1);
      if (aspect > 0.75 && aspect < 1.35) {
        return { shape: 'square', confidence: rectScore };
      }
      return { shape: 'rectangle', confidence: rectScore };
    }

    return null;
  }

  /**
   * Check if the stroke is approximately closed (start near end).
   */
  _isClosed(pts) {
    const first = pts[0];
    const last  = pts[pts.length - 1];
    const bb    = Utils.boundingBox(pts);
    const diag  = Math.sqrt(bb.w * bb.w + bb.h * bb.h) || 1;
    return Utils.dist(first, last) / diag < AppConfig.SHAPE_CLOSE_THRESHOLD;
  }

  /**
   * Measure how straight a stroke is using R² of linear regression.
   */
  _straightness(pts) {
    const n  = pts.length;
    let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
    for (const p of pts) {
      sx  += p.x;
      sy  += p.y;
      sxy += p.x * p.y;
      sx2 += p.x * p.x;
      sy2 += p.y * p.y;
    }
    const denom = Math.sqrt(
      (n * sx2 - sx * sx) * (n * sy2 - sy * sy)
    );
    if (denom === 0) return 1;
    const r = (n * sxy - sx * sy) / denom;
    return r * r;
  }

  /**
   * Compute circularity: 4π·Area / Perimeter²
   * Perfect circle = 1.0
   */
  _circularity(pts) {
    const perimeter = Utils.polylineLength(pts);
    if (perimeter === 0) return 0;
    const area = Math.abs(this._shoelaceArea(pts));
    return (4 * Math.PI * area) / (perimeter * perimeter);
  }

  /**
   * Shoelace formula for polygon area.
   */
  _shoelaceArea(pts) {
    let area = 0;
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += pts[i].x * pts[j].y;
      area -= pts[j].x * pts[i].y;
    }
    return area / 2;
  }

  /**
   * Score how rectangular a closed shape is (0-1).
   */
  _rectangleScore(pts) {
    const simplified = Utils.rdpSimplify(pts, 8);
    const corners    = simplified.length;
    if (corners < 4 || corners > 8) return 0;

    // Check that angles are approximately 90°
    let rightAngleScore = 0;
    let checked = 0;
    for (let i = 0; i < corners; i++) {
      const a = simplified[(i - 1 + corners) % corners];
      const b = simplified[i];
      const c = simplified[(i + 1) % corners];
      const angle = this._angleDeg(a, b, c);
      const diff  = Math.abs(angle - 90);
      if (diff < 35) { rightAngleScore += (1 - diff / 35); checked++; }
    }
    if (checked < 3) return 0;
    return (rightAngleScore / checked) * 0.85;
  }

  /**
   * Score how triangular a closed shape is (0-1).
   */
  _triangleScore(pts) {
    const simplified = Utils.rdpSimplify(pts, 10);
    if (simplified.length < 3 || simplified.length > 6) return 0;
    let angleSum = 0;
    let n = simplified.length;
    for (let i = 0; i < n; i++) {
      const a = simplified[(i - 1 + n) % n];
      const b = simplified[i];
      const c = simplified[(i + 1) % n];
      const angle = this._angleDeg(a, b, c);
      angleSum += angle;
    }
    // Sum of angles in a triangle ≈ 180°
    const targetSum = 180 * (n - 2);
    const ratio = Math.min(angleSum, targetSum) / Math.max(angleSum, targetSum);
    return ratio * 0.9;
  }

  /**
   * Check if the end of a stroke has an arrowhead (rapid direction change).
   */
  _hasArrowhead(pts) {
    if (pts.length < 8) return false;
    const tailLen   = Math.min(8, Math.floor(pts.length * 0.25));
    const mainPts   = pts.slice(0, pts.length - tailLen);
    const arrowPts  = pts.slice(pts.length - tailLen);

    // Direction of main stroke
    const mainDir = {
      x: mainPts[mainPts.length - 1].x - mainPts[0].x,
      y: mainPts[mainPts.length - 1].y - mainPts[0].y,
    };
    // Direction of arrow tip
    const tipDir = {
      x: arrowPts[arrowPts.length - 1].x - arrowPts[0].x,
      y: arrowPts[arrowPts.length - 1].y - arrowPts[0].y,
    };

    const dot  = mainDir.x * tipDir.x + mainDir.y * tipDir.y;
    const magA = Math.sqrt(mainDir.x ** 2 + mainDir.y ** 2);
    const magB = Math.sqrt(tipDir.x ** 2 + tipDir.y ** 2);
    if (!magA || !magB) return false;
    const cosAngle = dot / (magA * magB);
    // If the tip direction diverges significantly from main direction
    return cosAngle < 0.6;
  }

  /**
   * Compute the interior angle at point B formed by A-B-C, in degrees.
   */
  _angleDeg(a, b, c) {
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot  = ba.x * bc.x + ba.y * bc.y;
    const magA = Math.sqrt(ba.x ** 2 + ba.y ** 2);
    const magB = Math.sqrt(bc.x ** 2 + bc.y ** 2);
    if (!magA || !magB) return 0;
    const cos = Utils.clamp(dot / (magA * magB), -1, 1);
    return Math.acos(cos) * (180 / Math.PI);
  }

  /**
   * Render a clean geometric shape onto the drawing canvas.
   * @param {CanvasRenderingContext2D} ctx
   * @param {string} shape
   * @param {Array<{x,y}>} pts
   * @param {BrushEngine} brush
   */
  renderCleanShape(ctx, shape, pts, brush) {
    const bb = Utils.boundingBox(pts);
    const cx = (bb.minX + bb.maxX) / 2;
    const cy = (bb.minY + bb.maxY) / 2;

    const cfg = AppConfig.BRUSH_CONFIGS[brush.tool] || AppConfig.BRUSH_CONFIGS.pencil;
    ctx.save();
    ctx.globalCompositeOperation = cfg.composite;
    ctx.globalAlpha  = Utils.clamp(brush.opacity * cfg.globalAlpha, 0, 1);
    ctx.strokeStyle  = brush.color;
    ctx.lineWidth    = brush.size * cfg.lineWidthMultiplier;
    ctx.lineCap      = 'round';
    ctx.lineJoin     = 'round';

    if (cfg.glow) {
      ctx.shadowColor = brush.color;
      ctx.shadowBlur  = brush.size * 3;
    }

    ctx.beginPath();

    switch (shape) {
      case 'circle': {
        const rx = bb.w / 2;
        const ry = bb.h / 2;
        const r  = (rx + ry) / 2; // average for "clean" circle
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        break;
      }
      case 'rectangle':
        ctx.rect(bb.minX, bb.minY, bb.w, bb.h);
        break;
      case 'square': {
        const side = (bb.w + bb.h) / 2;
        ctx.rect(cx - side / 2, cy - side / 2, side, side);
        break;
      }
      case 'triangle': {
        // Equilateral-ish triangle
        ctx.moveTo(cx, bb.minY);
        ctx.lineTo(bb.maxX, bb.maxY);
        ctx.lineTo(bb.minX, bb.maxY);
        ctx.closePath();
        break;
      }
      case 'line': {
        ctx.moveTo(pts[0].x, pts[0].y);
        ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
        break;
      }
      case 'arrow': {
        const start = pts[0];
        const end   = pts[pts.length - 1];
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const nx = dx / len;
        const ny = dy / len;
        const arrowSize = Math.max(brush.size * 3, 18);
        const perpX = -ny;
        const perpY =  nx;
        // Shaft
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        // Arrowhead
        ctx.moveTo(end.x, end.y);
        ctx.lineTo(end.x - arrowSize * nx + arrowSize * 0.4 * perpX,
                   end.y - arrowSize * ny + arrowSize * 0.4 * perpY);
        ctx.moveTo(end.x, end.y);
        ctx.lineTo(end.x - arrowSize * nx - arrowSize * 0.4 * perpX,
                   end.y - arrowSize * ny - arrowSize * 0.4 * perpY);
        break;
      }
    }

    ctx.stroke();
    ctx.restore();
  }
        }
        


/* ═══════════════════════════════════════════════════════
   8. GESTURE DETECTOR
═══════════════════════════════════════════════════════ */
class GestureDetector {
  /**
   * MediaPipe landmark indices:
   * 0=WRIST 1=THUMB_CMC 2=THUMB_MCP 3=THUMB_IP 4=THUMB_TIP
   * 5=INDEX_MCP 6=INDEX_PIP 7=INDEX_DIP 8=INDEX_TIP
   * 9=MIDDLE_MCP 10=MIDDLE_PIP 11=MIDDLE_DIP 12=MIDDLE_TIP
   * 13=RING_MCP 14=RING_PIP 15=RING_DIP 16=RING_TIP
   * 17=PINKY_MCP 18=PINKY_PIP 19=PINKY_DIP 20=PINKY_TIP
   */
  static WRIST       = 0;
  static INDEX_MCP   = 5;
  static INDEX_TIP   = 8;
  static MIDDLE_MCP  = 9;
  static MIDDLE_TIP  = 12;
  static RING_MCP    = 13;
  static RING_TIP    = 16;
  static PINKY_MCP   = 17;
  static PINKY_TIP   = 20;
  static THUMB_TIP   = 4;
  static THUMB_MCP   = 2;

  /**
   * Classify landmarks into one of: 'draw', 'pause', 'fist', 'pinch', 'unknown'
   * @param {Array} landmarks  normalized landmarks [{x,y,z},…]
   * @returns {string}
   */
  classify(landmarks) {
    if (!landmarks || landmarks.length < 21) return 'unknown';

    const wrist  = landmarks[GestureDetector.WRIST];
    const iTip   = landmarks[GestureDetector.INDEX_TIP];
    const mTip   = landmarks[GestureDetector.MIDDLE_TIP];
    const rTip   = landmarks[GestureDetector.RING_TIP];
    const pTip   = landmarks[GestureDetector.PINKY_TIP];
    const tTip   = landmarks[GestureDetector.THUMB_TIP];
    const iMcp   = landmarks[GestureDetector.INDEX_MCP];
    const mMcp   = landmarks[GestureDetector.MIDDLE_MCP];
    const rMcp   = landmarks[GestureDetector.RING_MCP];
    const pMcp   = landmarks[GestureDetector.PINKY_MCP];

    // Check extension of each finger (tip above mcp in screen-y = extended)
    const indexExtended  = this._fingerExtended(landmarks, 5,  8);
    const middleExtended = this._fingerExtended(landmarks, 9,  12);
    const ringExtended   = this._fingerExtended(landmarks, 13, 16);
    const pinkyExtended  = this._fingerExtended(landmarks, 17, 20);

    // Pinch: thumb tip close to index tip
    const pinchDist = Utils.dist(tTip, iTip);
    if (pinchDist < AppConfig.PINCH_THRESHOLD) {
      return 'pinch';
    }

    // Draw: only index finger up, others folded
    if (indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return 'draw';
    }

    // Open palm: all fingers extended
    if (indexExtended && middleExtended && ringExtended && pinkyExtended) {
      return 'pause';
    }

    // Fist: no fingers extended
    if (!indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
      return 'fist';
    }

    return 'unknown';
  }

  /**
   * Check if a finger is extended.
   * @param {Array} lm  all landmarks
   * @param {number} mcpIdx  MCP joint index
   * @param {number} tipIdx  TIP landmark index
   */
  _fingerExtended(lm, mcpIdx, tipIdx) {
    const tip = lm[tipIdx];
    const mcp = lm[mcpIdx];
    // Tip is above (lower y in screen space) the MCP by a meaningful amount
    return (mcp.y - tip.y) > 0.04;
  }

  /**
   * Get the index fingertip position in canvas space.
   * @param {Array} landmarks normalized landmarks
   * @param {number} canvasW
   * @param {number} canvasH
   * @returns {{x:number, y:number}}
   */
  getIndexTip(landmarks, canvasW, canvasH) {
    if (!landmarks || landmarks.length < 9) return null;
    const tip = landmarks[GestureDetector.INDEX_TIP];
    // Mirror X because camera is mirrored
    return {
      x: (1 - tip.x) * canvasW,
      y: tip.y * canvasH,
    };
  }
}

/* ═══════════════════════════════════════════════════════
   9. CAMERA MANAGER
═══════════════════════════════════════════════════════ */
class CameraManager {
  constructor(videoEl) {
    this._videoEl   = videoEl;
    this._stream    = null;
    this._facingMode = 'user'; // front camera
    this.isDemoMode  = false;
  }

  /**
   * Request camera access and start stream.
   * @returns {Promise<void>}
   */
  async start() {
    const constraints = {
      video: {
        facingMode: this._facingMode,
        width:  { ideal: AppConfig.IDEAL_WIDTH },
        height: { ideal: AppConfig.IDEAL_HEIGHT },
        frameRate: { ideal: AppConfig.IDEAL_FPS },
      },
      audio: false,
    };

    try {
      this._stream = await navigator.mediaDevices.getUserMedia(constraints);
      this._videoEl.srcObject = this._stream;
      await new Promise((resolve, reject) => {
        this._videoEl.onloadedmetadata = resolve;
        this._videoEl.onerror = reject;
        setTimeout(reject, 8000, new Error('Video metadata timeout'));
      });
      await this._videoEl.play();
    } catch (err) {
      this._handleCameraError(err);
      throw err;
    }
  }

  /**
   * Switch between front and back cameras.
   */
  async switchCamera() {
    await this.stop();
    this._facingMode = this._facingMode === 'user' ? 'environment' : 'user';
    await this.start();
  }

  /**
   * Stop the camera stream.
   */
  async stop() {
    if (this._stream) {
      for (const track of this._stream.getTracks()) track.stop();
      this._stream = null;
      this._videoEl.srcObject = null;
    }
  }

  /**
   * Parse camera errors into friendly messages.
   */
  _handleCameraError(err) {
    let title, message;
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      title   = 'Camera Access Denied';
      message = 'Please allow camera access in your browser settings, then reload the page.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      title   = 'No Camera Found';
      message = 'No camera device was found. Connect a camera and try again.';
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      title   = 'Camera In Use';
      message = 'Your camera is being used by another application. Close it and try again.';
    } else if (err.name === 'OverconstrainedError') {
      title   = 'Camera Unsupported';
      message = 'Your camera does not support the required resolution. Trying with default settings…';
    } else {
      title   = 'Camera Error';
      message = err.message || 'An unknown camera error occurred.';
    }
    console.error('[CameraManager]', err);
    this._lastError = { title, message };
  }

  get lastError() { return this._lastError || null; }

  get isActive() {
    return this._stream !== null && this._stream.active;
  }

  get facingMode() { return this._facingMode; }

  get videoEl() { return this._videoEl; }
}

/* ═══════════════════════════════════════════════════════
   10. HAND TRACKER
═══════════════════════════════════════════════════════ */
class HandTracker {
  /**
   * @param {HTMLVideoElement} videoEl
   * @param {function} onResults  called with MediaPipe results
   */
  constructor(videoEl, onResults) {
    this._videoEl   = videoEl;
    this._onResults = onResults;
    this._hands     = null;
    this._camera    = null;
    this._running   = false;
    this._lastLandmarks = null;
  }

  /**
   * Initialize MediaPipe Hands and start processing.
   */
  async init() {
    return new Promise((resolve, reject) => {
      try {
        this._hands = new Hands({
          locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        this._hands.setOptions({
          maxNumHands:             AppConfig.MEDIAPIPE_MAX_NUM_HANDS,
          modelComplexity:         AppConfig.MEDIAPIPE_MODEL_COMPLEXITY,
          minDetectionConfidence:  AppConfig.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
          minTrackingConfidence:   AppConfig.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        });

        this._hands.onResults((results) => {
          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            this._lastLandmarks = results.multiHandLandmarks[0];
          } else {
            this._lastLandmarks = null;
          }
          this._onResults(results);
        });

        // Use MediaPipe Camera utility to drive the hand tracking loop
        this._camera = new Camera(this._videoEl, {
          onFrame: async () => {
            if (this._running && this._hands) {
              try {
                await this._hands.send({ image: this._videoEl });
              } catch (e) {
                // Silently ignore frame send errors (stream stopped mid-send)
              }
            }
          },
          width:  AppConfig.IDEAL_WIDTH,
          height: AppConfig.IDEAL_HEIGHT,
        });

        this._running = true;
        this._camera.start()
          .then(() => resolve())
          .catch(reject);
      } catch (err) {
        reject(err);
      }
    });
  }

  /**
   * Stop hand tracking.
   */
  async stop() {
    this._running = false;
    if (this._camera) {
      try { await this._camera.stop(); } catch (_) {}
      this._camera = null;
    }
    if (this._hands) {
      try { this._hands.close(); } catch (_) {}
      this._hands = null;
    }
  }

  get lastLandmarks() { return this._lastLandmarks; }
}


/* ═══════════════════════════════════════════════════════
   11. UI CONTROLLER
═══════════════════════════════════════════════════════ */
class UIController {
  constructor() {
    // Screen elements
    this.loadingScreen    = document.getElementById('loading-screen');
    this.permissionScreen = document.getElementById('permission-screen');
    this.errorScreen      = document.getElementById('error-screen');
    this.appScreen        = document.getElementById('app-screen');

    // Loading
    this.loadingBar  = document.getElementById('loading-bar');
    this.loadingStep = document.getElementById('loading-step');

    // HUD
    this.gestureIcon     = document.getElementById('gesture-icon');
    this.gestureLabel    = document.getElementById('gesture-label');
    this.gestureDisplay  = document.querySelector('.gesture-display');
    this.fpsValue        = document.getElementById('fps-value');
    this.confidenceValue = document.getElementById('confidence-value');

    // Toolbar — brush tools
    this.toolBtns   = document.querySelectorAll('.tool-btn[data-tool]');
    this.sizeSlider = document.getElementById('brush-size-slider');
    this.opacitySlider = document.getElementById('opacity-slider');
    this.sizeLbl    = document.getElementById('size-value-label');
    this.opacityLbl = document.getElementById('opacity-value-label');

    // Toolbar — action buttons
    this.undoBtn     = document.getElementById('undo-btn');
    this.redoBtn     = document.getElementById('redo-btn');
    this.clearBtn    = document.getElementById('clear-btn');
    this.saveBtn     = document.getElementById('save-btn');
    this.shapeToggle = document.getElementById('shape-recognition-btn');

    // Color controls
    this.colorGrid        = document.getElementById('color-grid');
    this.customColorInput = document.getElementById('custom-color-input');

    // Camera controls
    this.cameraSwitchBtn  = document.getElementById('camera-switch-btn');
    this.requestCameraBtn = document.getElementById('request-camera-btn');
    this.demoBtnPerm      = document.getElementById('demo-mode-btn');
    this.demoBtnError     = document.getElementById('demo-from-error-btn');
    this.retryBtn         = document.getElementById('retry-btn');

    // Debug
    this.debugToggleBtn = document.getElementById('debug-toggle-btn');
    this.debugPanel     = document.getElementById('debug-panel');
    this.debugCloseBtn  = document.getElementById('debug-close-btn');
    this.dbgGesture     = document.getElementById('dbg-gesture');
    this.dbgDrawing     = document.getElementById('dbg-drawing');
    this.dbgFps         = document.getElementById('dbg-fps');
    this.dbgPos         = document.getElementById('dbg-pos');
    this.dbgPts         = document.getElementById('dbg-pts');
    this.dbgHistory     = document.getElementById('dbg-history');
    this.dbgLandmarks   = document.getElementById('dbg-landmarks');
    this.dbgCamera      = document.getElementById('dbg-camera');

    // Error screen
    this.errorTitle   = document.getElementById('error-title');
    this.errorMessage = document.getElementById('error-message');

    // Toast elements
    this.shapeToast        = document.getElementById('shape-toast');
    this.notificationToast = document.getElementById('notification-toast');
    this._toastTimer       = null;
    this._shapeToastTimer  = null;

    // Build color palette
    this._buildColorGrid();
  }

  /* ── SCREENS ─────────────────────────────────────── */

  showLoading() {
    this._setScreen('loading-screen');
  }

  showPermission() {
    this._setScreen('permission-screen');
  }

  showError(title, message) {
    this.errorTitle.textContent   = title   || 'Error';
    this.errorMessage.textContent = message || 'An unexpected error occurred.';
    this._setScreen('error-screen');
  }

  showApp() {
    this._setScreen('app-screen');
    this.appScreen.classList.add('active');
  }

  _setScreen(id) {
    for (const el of document.querySelectorAll('.screen')) {
      el.classList.toggle('active', el.id === id);
    }
    // App screen has a special class
    const appActive = id === 'app-screen';
    this.appScreen.style.display = appActive ? 'block' : 'none';
    if (appActive) this.appScreen.classList.add('active');
  }

  /* ── LOADING PROGRESS ────────────────────────────── */

  setLoadingProgress(pct, stepText) {
    this.loadingBar.style.width = `${Utils.clamp(pct, 0, 100)}%`;
    if (stepText) this.loadingStep.textContent = stepText;
  }

  /* ── GESTURE HUD ─────────────────────────────────── */

  updateGesture(gesture, confidence) {
    const labels = {
      draw:    { icon: '☝️',  text: 'Drawing',   cls: 'draw' },
      pause:   { icon: '🖐️', text: 'Paused',     cls: 'pause' },
      fist:    { icon: '✊',  text: 'Fist — Hold to Clear', cls: 'fist' },
      pinch:   { icon: '🤌',  text: 'Eraser Mode', cls: 'pinch' },
      unknown: { icon: '✋',  text: 'Waiting…',   cls: '' },
    };
    const info = labels[gesture] || labels.unknown;
    this.gestureIcon.textContent  = info.icon;
    this.gestureLabel.textContent = info.text;
    this.gestureDisplay.className = 'gesture-display' + (info.cls ? ` ${info.cls}` : '');
    if (confidence !== undefined) {
      this.confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
    }
  }

  updateFps(fps) {
    this.fpsValue.textContent = fps.toFixed(0);
    if (this.debugPanel && !this.debugPanel.hidden) {
      this.dbgFps.textContent = fps.toFixed(1);
    }
  }

  /* ── DEBUG PANEL ─────────────────────────────────── */

  updateDebug(data) {
    if (!this.debugPanel || this.debugPanel.hidden) return;
    if (data.gesture   !== undefined) this.dbgGesture.textContent   = data.gesture;
    if (data.drawing   !== undefined) this.dbgDrawing.textContent   = data.drawing;
    if (data.pos       !== undefined) this.dbgPos.textContent       = data.pos;
    if (data.pts       !== undefined) this.dbgPts.textContent       = data.pts;
    if (data.history   !== undefined) this.dbgHistory.textContent   = data.history;
    if (data.landmarks !== undefined) this.dbgLandmarks.textContent = data.landmarks;
    if (data.camera    !== undefined) this.dbgCamera.textContent    = data.camera;
  }

  toggleDebug() {
    this.debugPanel.hidden = !this.debugPanel.hidden;
    this.debugToggleBtn.classList.toggle('active', !this.debugPanel.hidden);
  }

  /* ── COLOR PALETTE ───────────────────────────────── */

  _buildColorGrid() {
    this.colorGrid.innerHTML = '';
    AppConfig.PALETTE_COLORS.forEach((color, i) => {
      const swatch = document.createElement('button');
      swatch.className       = 'color-swatch' + (i === 13 ? ' selected' : ''); // violet default
      swatch.style.background = color;
      swatch.title            = color;
      swatch.setAttribute('aria-label', `Color ${color}`);
      swatch.setAttribute('role', 'radio');
      swatch.setAttribute('aria-checked', i === 13 ? 'true' : 'false');
      swatch.dataset.color = color;
      this.colorGrid.appendChild(swatch);
    });
  }

  /**
   * Highlight the selected color swatch.
   */
  selectColorSwatch(color) {
    for (const sw of this.colorGrid.querySelectorAll('.color-swatch')) {
      const selected = sw.dataset.color === color;
      sw.classList.toggle('selected', selected);
      sw.setAttribute('aria-checked', selected ? 'true' : 'false');
    }
  }

  /* ── TOOL BUTTONS ────────────────────────────────── */

  selectTool(toolName) {
    for (const btn of this.toolBtns) {
      const active = btn.dataset.tool === toolName;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
  }

  /* ── SLIDERS ─────────────────────────────────────── */

  updateSizeLabel(val) {
    this.sizeLbl.textContent = val;
  }

  updateOpacityLabel(val) {
    this.opacityLbl.textContent = `${val}%`;
  }

  /* ── TOASTS ──────────────────────────────────────── */

  showShapeToast(shapeName) {
    const names = {
      circle:    '⭕ Circle recognized',
      rectangle: '▭ Rectangle recognized',
      square:    '□ Square recognized',
      triangle:  '△ Triangle recognized',
      line:      '— Line recognized',
      arrow:     '→ Arrow recognized',
    };
    this.shapeToast.textContent = names[shapeName] || shapeName;
    this.shapeToast.classList.add('visible');
    clearTimeout(this._shapeToastTimer);
    this._shapeToastTimer = setTimeout(() => {
      this.shapeToast.classList.remove('visible');
    }, AppConfig.SHAPE_TOAST_DURATION_MS);
  }

  showNotification(message, type = '') {
    this.notificationToast.textContent = message;
    this.notificationToast.className   = 'notification-toast visible' + (type ? ` ${type}` : '');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      this.notificationToast.classList.remove('visible');
    }, AppConfig.TOAST_DURATION_MS);
  }
}

/* ═══════════════════════════════════════════════════════
   12. COLOR PALETTE MANAGER
═══════════════════════════════════════════════════════ */
class ColorPaletteManager {
  constructor(ui, onChange) {
    this._ui        = ui;
    this._onChange  = onChange;
    this.color      = AppConfig.DEFAULT_COLOR;
    this._bindEvents();
  }

  _bindEvents() {
    // Swatch clicks
    this._ui.colorGrid.addEventListener('click', (e) => {
      const swatch = e.target.closest('.color-swatch');
      if (!swatch) return;
      this.setColor(swatch.dataset.color);
      this._ui.selectColorSwatch(this.color);
    });

    // Custom color picker
    this._ui.customColorInput.addEventListener('input', (e) => {
      this.setColor(e.target.value);
      this._ui.selectColorSwatch(''); // deselect presets
    });
  }

  setColor(hex) {
    this.color = hex;
    this._onChange(hex);
  }
}

/* ═══════════════════════════════════════════════════════
   13. FPS COUNTER
═══════════════════════════════════════════════════════ */
class FpsCounter {
  constructor() {
    this._frames     = 0;
    this._lastUpdate = performance.now();
    this.fps         = 0;
  }

  tick() {
    this._frames++;
    const now  = performance.now();
    const diff = now - this._lastUpdate;
    if (diff >= AppConfig.FPS_UPDATE_INTERVAL) {
      this.fps         = (this._frames / diff) * 1000;
      this._frames     = 0;
      this._lastUpdate = now;
      return true; // updated
    }
    return false;
  }
}

/* ═══════════════════════════════════════════════════════
   14. OVERLAY RENDERER (landmark + cursor visualization)
═══════════════════════════════════════════════════════ */
class OverlayRenderer {
  constructor(canvasManager) {
    this._cm  = canvasManager;
    this._ctx = canvasManager.overlayCtx;
  }

  /**
   * Draw the finger cursor dot + optional landmark skeleton.
   */
  render(fingerPos, gesture, showSkeleton, landmarks) {
    const ctx = this._ctx;
    ctx.clearRect(0, 0, this._cm.width, this._cm.height);

    if (!fingerPos) return;

    const colors = {
      draw:    '#34d399',
      pause:   '#fbbf24',
      fist:    '#fb7185',
      pinch:   '#38bdf8',
      unknown: 'rgba(255,255,255,0.4)',
    };
    const color = colors[gesture] || colors.unknown;

    // Cursor dot
    ctx.save();
    ctx.beginPath();
    ctx.arc(fingerPos.x, fingerPos.y, 8, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 16;
    ctx.fill();

    // Inner white dot
    ctx.beginPath();
    ctx.arc(fingerPos.x, fingerPos.y, 3, 0, Math.PI * 2);
    ctx.fillStyle   = 'white';
    ctx.shadowBlur  = 0;
    ctx.fill();
    ctx.restore();

    // Draw mode: show trailing ripple
    if (gesture === 'draw') {
      ctx.save();
      ctx.beginPath();
      ctx.arc(fingerPos.x, fingerPos.y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth   = 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // Skeleton (debug mode)
    if (showSkeleton && landmarks) {
      this._drawSkeleton(ctx, landmarks);
    }
  }

  /**
   * Draw a simplified hand skeleton for debug mode.
   */
  _drawSkeleton(ctx, landmarks) {
    const W = this._cm.width;
    const H = this._cm.height;

    const pt = (idx) => ({
      x: (1 - landmarks[idx].x) * W,
      y: landmarks[idx].y * H,
    });

    const connections = [
      [0,1],[1,2],[2,3],[3,4],   // Thumb
      [0,5],[5,6],[6,7],[7,8],   // Index
      [5,9],[9,10],[10,11],[11,12],  // Middle
      [9,13],[13,14],[14,15],[15,16], // Ring
      [13,17],[17,18],[18,19],[19,20], // Pinky
      [0,17],
    ];

    ctx.save();
    ctx.globalAlpha = 0.45;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth   = 1.5;

    for (const [a, b] of connections) {
      const pa = pt(a);
      const pb = pt(b);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }

    // Landmark dots
    ctx.fillStyle = '#a78bfa';
    for (let i = 0; i < 21; i++) {
      const p = pt(i);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }
  }
 
/* ═══════════════════════════════════════════════════════
   15. AIR DRAW APP (Orchestrator)
═══════════════════════════════════════════════════════ */
class AirDrawApp {
  constructor() {
    // Core subsystems
    this._ui           = new UIController();
    this._cm           = new CanvasManager();
    this._history      = new HistoryManager(this._cm);
    this._brush        = new BrushEngine(this._cm);
    this._stroke       = new StrokeBuffer();
    this._shapeRec     = new ShapeRecognizer();
    this._gestureDetec = new GestureDetector();
    this._overlay      = new OverlayRenderer(this._cm);
    this._fps          = new FpsCounter();
    this._palette      = new ColorPaletteManager(this._ui, (c) => this._onColorChange(c));

    // Camera & hand tracking
    this._camera     = new CameraManager(document.getElementById('input-video'));
    this._tracker    = null;

    // Application state
    this._gesture       = 'unknown';
    this._lastGesture   = 'unknown';
    this._isDrawing     = false;
    this._gestureStart  = 0;
    this._fistHoldTimer = null;
    this._fingerPos     = null;
    this._shapeEnabled  = true;
    this._debugMode     = false;
    this._demoMode      = false;
    this._demoAnimFrame = null;
    this._handConfidence = 0;
    this._hasHand       = false;
    this._cameraFrameId = null;

    // Shape recognition state
    this._shapeTimer    = null;
    this._pendingStroke = null; // points for shape recognition after draw ends

    // Gesture state machine cooldown
    this._gestureChangeTime = 0;

    this._bindUI();
  }

  /* ── PUBLIC: INIT ────────────────────────────────── */

  async init() {
    this._ui.showLoading();
    this._ui.setLoadingProgress(10, 'Loading MediaPipe…');

    await this._waitForMediaPipe();
    this._ui.setLoadingProgress(30, 'Checking camera permission…');

    const hasPerm = await this._checkCameraPermission();

    if (hasPerm) {
      this._ui.setLoadingProgress(55, 'Starting camera…');
      await this._startCameraAndTracking();
    } else {
      this._ui.setLoadingProgress(100, 'Ready');
      setTimeout(() => this._ui.showPermission(), 300);
    }
  }

  /* ── MEDIAPIPE LOAD WAIT ─────────────────────────── */

  _waitForMediaPipe() {
    return new Promise((resolve, reject) => {
      let attempts = 0;
      const check = () => {
        attempts++;
        if (typeof Hands !== 'undefined' && typeof Camera !== 'undefined') {
          resolve();
        } else if (attempts > 80) {
          reject(new Error('MediaPipe failed to load. Check your internet connection.'));
        } else {
          setTimeout(check, 150);
        }
      };
      check();
    });
  }

  /* ── CAMERA PERMISSION ───────────────────────────── */

  async _checkCameraPermission() {
    try {
      const perm = await navigator.permissions.query({ name: 'camera' });
      return perm.state === 'granted';
    } catch (_) {
      // permissions API not supported; attempt camera directly
      return false;
    }
  }

  /* ── CAMERA + TRACKING START ─────────────────────── */

  async _startCameraAndTracking() {
    try {
      this._ui.setLoadingProgress(60, 'Requesting camera access…');
      await this._camera.start();
      this._ui.setLoadingProgress(75, 'Initializing hand tracking…');

      this._tracker = new HandTracker(
        this._camera.videoEl,
        (results) => this._onHandResults(results)
      );

      await this._tracker.init();
      this._ui.setLoadingProgress(95, 'Almost ready…');
      await new Promise(r => setTimeout(r, 400));
      this._ui.setLoadingProgress(100, 'Launching…');
      await new Promise(r => setTimeout(r, 300));

      this._ui.showApp();
      this._startCameraRenderLoop();
      this._ui.updateDebug({ camera: this._camera.facingMode });

    } catch (err) {
      console.error('[AirDrawApp] Startup error:', err);
      const camErr = this._camera.lastError;
      const title  = camErr?.title   || 'Startup Failed';
      const msg    = camErr?.message || err.message || 'Unknown error during startup.';
      this._ui.showError(title, msg);
    }
  }

  /* ── CAMERA RENDER LOOP ──────────────────────────── */

  _startCameraRenderLoop() {
    const loop = () => {
      this._cm.drawCameraFrame(this._camera.videoEl);
      this._cameraFrameId = requestAnimationFrame(loop);
    };
    this._cameraFrameId = requestAnimationFrame(loop);
  }

  /* ── MEDIAPIPE RESULTS ───────────────────────────── */

  _onHandResults(results) {
    // FPS tick
    if (this._fps.tick()) {
      this._ui.updateFps(this._fps.fps);
    }

    const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    this._hasHand = hasHand;

    if (!hasHand) {
      this._handleNoHand();
      return;
    }

    const landmarks   = results.multiHandLandmarks[0];
    const worldLm     = results.multiHandWorldLandmarks?.[0];
    const confidence  = results.multiHandedness?.[0]?.score ?? 0;
    this._handConfidence = confidence;

    // Classify gesture
    const rawGesture = this._gestureDetec.classify(landmarks);

    // Apply temporal smoothing to prevent gesture flicker
    const now = performance.now();
    if (rawGesture !== this._lastGesture) {
      this._gestureChangeTime = now;
    }
    const gestureDwellMs = now - this._gestureChangeTime;
    const stableGesture  = (gestureDwellMs > AppConfig.DRAW_COOLDOWN_MS)
      ? rawGesture : this._gesture;
    this._lastGesture = rawGesture;

    // If gesture changed (after dwell), process transition
    if (stableGesture !== this._gesture) {
      this._onGestureChange(this._gesture, stableGesture);
      this._gesture = stableGesture;
    }

    // Get finger position in canvas space
    const pos = this._gestureDetec.getIndexTip(
      landmarks, this._cm.width, this._cm.height
    );
    this._fingerPos = pos;

    // Process drawing based on gesture
    this._processGestureFrame(stableGesture, pos);

    // Update overlay (cursor + optional skeleton)
    this._overlay.render(pos, stableGesture, this._debugMode, landmarks);

    // Update HUD
    this._ui.updateGesture(stableGesture, confidence);
    this._ui.updateDebug({
      gesture:   stableGesture,
      drawing:   this._isDrawing ? 'YES' : 'no',
      pos:       pos ? `${pos.x.toFixed(0)}, ${pos.y.toFixed(0)}` : '--',
      pts:       this._stroke.length,
      history:   this._history.stackInfo,
      landmarks: `${landmarks.length} pts`,
    });
  }

  /* ── NO HAND DETECTED ────────────────────────────── */

  _handleNoHand() {
    // If we were drawing, commit the stroke
    if (this._isDrawing) {
      this._commitStroke();
    }
    this._gesture   = 'unknown';
    this._lastGesture = 'unknown';
    this._fingerPos = null;
    this._isDrawing = false;
    this._overlay.render(null, 'unknown', false, null);
    this._ui.updateGesture('unknown', 0);
    this._ui.updateDebug({ gesture: 'unknown', drawing: 'no', landmarks: '0 pts' });
  }

  /* ── GESTURE STATE TRANSITION ────────────────────── */

  _onGestureChange(from, to) {
    // When we STOP drawing, commit the stroke
    if (from === 'draw' && to !== 'draw') {
      this._commitStroke();
    }

    // When entering draw from non-draw, start fresh stroke
    if (from !== 'draw' && to === 'draw') {
      this._stroke.reset();
      this._isDrawing = false; // will be set to true when first point added
    }

    // Pinch = eraser mode
    if (to === 'pinch') {
      this._prePinchTool = this._brush.tool !== 'eraser' ? this._brush.tool : this._prePinchTool;
      this._ui.selectTool('eraser');
      this._brush.tool = 'eraser';
      this._ui.showNotification('Eraser mode (pinch to exit)', '');
    } else if (from === 'pinch' && to !== 'pinch') {
      // Restore previous tool
      if (this._prePinchTool && this._prePinchTool !== 'eraser') {
        this._brush.tool = this._prePinchTool;
        this._ui.selectTool(this._prePinchTool);
      }
    }

    // Fist = hold to clear (handled in frame loop with timer)
    if (to === 'fist') {
      this._gestureStart = performance.now();
    } else {
      clearTimeout(this._fistHoldTimer);
      this._fistHoldTimer = null;
    }
  }

  /* ── PER-FRAME GESTURE PROCESSING ───────────────── */

  _processGestureFrame(gesture, pos) {
    switch (gesture) {
      case 'draw':
      case 'pinch':
        if (pos) this._addDrawPoint(pos.x, pos.y);
        break;

      case 'fist': {
        // Hold fist for 800ms to clear
        const held = performance.now() - this._gestureStart;
        if (held > 800 && !this._fistHoldTimer) {
          this._fistHoldTimer = setTimeout(() => {
            this._clearCanvas();
            this._ui.showNotification('Canvas cleared', 'success');
          }, 0);
        }
        break;
      }

      case 'pause':
      case 'unknown':
      default:
        break;
    }
  }

  /* ── DRAWING ─────────────────────────────────────── */

  _addDrawPoint(x, y) {
    const smoothPt = this._stroke.addPoint(x, y);
    if (!smoothPt) return;

    if (!this._isDrawing) {
      // First point — push pre-draw state to history
      this._history.push();
      this._isDrawing = true;
    }

    // Render the incremental segment onto the drawing canvas
    const tail = this._stroke.tail(4);
    this._brush.renderSegment(this._cm.drawingCtx, tail);
  }

  /**
   * Commit the current stroke: optionally run shape recognition.
   */
  _commitStroke() {
    if (!this._isDrawing || this._stroke.isEmpty) {
      this._stroke.reset();
      this._isDrawing = false;
      return;
    }

    const pts = this._stroke.points.slice();
    this._stroke.reset();
    this._isDrawing = false;

    // Schedule shape recognition if enabled
    if (this._shapeEnabled && this._brush.tool !== 'eraser' && pts.length >= AppConfig.SHAPE_MIN_POINTS) {
      clearTimeout(this._shapeTimer);
      this._pendingStroke = pts;
      this._shapeTimer = setTimeout(() => {
        this._tryShapeRecognition(pts);
      }, AppConfig.SHAPE_DETECTION_DELAY_MS);
    }
  }

  _tryShapeRecognition(pts) {
    const result = this._shapeRec.recognize(pts);
    if (!result) return;

    // Replace last stroke with clean geometric shape
    // First, revert to history snapshot from before the stroke
    // (history was pushed when stroke began, so undo to get pre-stroke state)
    const snap = this._cm.getDrawingSnapshot();
    this._history.undo();

    // Render the clean shape
    this._shapeRec.renderCleanShape(
      this._cm.drawingCtx,
      result.shape,
      pts,
      this._brush
    );

    // Push the clean shape into history
    this._history.push();

    this._ui.showShapeToast(result.shape);
  }

  /* ── CANVAS CLEAR ────────────────────────────────── */

  _clearCanvas() {
    this._history.push();
    this._cm.clearDrawing();
    this._stroke.reset();
    this._isDrawing = false;
  }

  /* ── COLOR CHANGE ────────────────────────────────── */

  _onColorChange(color) {
    this._brush.color = color;
  }

  /* ── DEMO MODE ───────────────────────────────────── */

  _startDemoMode() {
    this._demoMode = true;
    this._camera.isDemoMode = true;
    this._ui.showApp();

    // Fill the camera canvas with a gradient background
    const ctx = this._cm.cameraCtx;
    const grad = ctx.createLinearGradient(0, 0, this._cm.width, this._cm.height);
    grad.addColorStop(0, '#0a0a1a');
    grad.addColorStop(1, '#12122a');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this._cm.width, this._cm.height);

    this._ui.updateGesture('unknown', 0);
    this._ui.showNotification(
      'Demo mode — use toolbar buttons or keyboard shortcuts to draw',
      'warning'
    );

    // Draw demo text on camera canvas
    const cctx = this._cm.cameraCtx;
    cctx.fillStyle = 'rgba(167, 139, 250, 0.08)';
    cctx.font = `${Math.min(this._cm.width / 12, 48)}px system-ui`;
    cctx.textAlign = 'center';
    cctx.fillText('No Camera — Demo Mode', this._cm.width / 2, this._cm.height / 2);
    cctx.font = `${Math.min(this._cm.width / 22, 22)}px system-ui`;
    cctx.fillStyle = 'rgba(167, 139, 250, 0.04)';
    cctx.fillText('Use the toolbar on the left to draw with your mouse / touch',
      this._cm.width / 2, this._cm.height / 2 + 48);

    // Enable mouse/touch drawing on overlay canvas
    this._bindMouseDrawing();
  }

  /* ── MOUSE / TOUCH DRAWING (Demo + keyboard) ─────── */

  _bindMouseDrawing() {
    const canvas = this._cm.overlayCanvas;
    let mouseDown = false;

    const onStart = (e) => {
      e.preventDefault();
      mouseDown = true;
      this._history.push();
      this._isDrawing = true;
      this._stroke.reset();
      const pt = this._getCanvasPoint(e, canvas);
      this._stroke.addPoint(pt.x, pt.y);
    };

    const onMove = (e) => {
      e.preventDefault();
      if (!mouseDown) return;
      const pt = this._getCanvasPoint(e, canvas);
      const sp = this._stroke.addPoint(pt.x, pt.y);
      if (sp) {
        const tail = this._stroke.tail(4);
        this._brush.renderSegment(this._cm.drawingCtx, tail);
      }
    };

    const onEnd = (e) => {
      e.preventDefault();
      if (!mouseDown) return;
      mouseDown = false;
      this._commitStroke();
    };

    canvas.addEventListener('mousedown',  onStart, { passive: false });
    canvas.addEventListener('mousemove',  onMove,  { passive: false });
    canvas.addEventListener('mouseup',    onEnd,   { passive: false });
    canvas.addEventListener('mouseleave', onEnd,   { passive: false });

    canvas.addEventListener('touchstart', onStart, { passive: false });
    canvas.addEventListener('touchmove',  onMove,  { passive: false });
    canvas.addEventListener('touchend',   onEnd,   { passive: false });
    canvas.addEventListener('touchcancel',onEnd,   { passive: false });
  }

  _getCanvasPoint(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches && e.touches.length > 0) {
      const t = e.touches[0];
      return {
        x: (t.clientX - rect.left) * scaleX,
        y: (t.clientY - rect.top)  * scaleY,
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    };
                   }
    /* ── UI BINDINGS ─────────────────────────────────── */

  _bindUI() {
    const ui = this._ui;

    /* Permission screen */
    ui.requestCameraBtn.addEventListener('click', async () => {
      ui.showLoading();
      ui.setLoadingProgress(50, 'Starting camera…');
      await this._startCameraAndTracking();
    });

    ui.demoBtnPerm.addEventListener('click', () => {
      this._startDemoMode();
    });

    ui.demoBtnError.addEventListener('click', () => {
      this._startDemoMode();
    });

    ui.retryBtn.addEventListener('click', async () => {
      ui.showLoading();
      ui.setLoadingProgress(30, 'Retrying…');
      await this._startCameraAndTracking();
    });

    /* Camera switch */
    ui.cameraSwitchBtn.addEventListener('click', async () => {
      if (this._demoMode) return;
      try {
        ui.showNotification('Switching camera…', '');
        await this._camera.switchCamera();
        if (this._tracker) {
          await this._tracker.stop();
        }
        this._tracker = new HandTracker(
          this._camera.videoEl,
          (results) => this._onHandResults(results)
        );
        await this._tracker.init();
        ui.showNotification('Camera switched', 'success');
        ui.updateDebug({ camera: this._camera.facingMode });
      } catch (err) {
        ui.showNotification('Failed to switch camera', 'error');
        console.error('[CameraSwitch]', err);
      }
    });

    /* Tool buttons */
    ui.toolBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const tool = btn.dataset.tool;
        this._brush.tool = tool;
        ui.selectTool(tool);
        if (tool === 'eraser') {
          this._prePinchTool = null;
        }
      });
    });

    /* Brush size */
    ui.sizeSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      this._brush.size = val;
      ui.updateSizeLabel(val);
    });

    /* Opacity */
    ui.opacitySlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      this._brush.opacity = val / 100;
      ui.updateOpacityLabel(val);
    });

    /* Undo */
    ui.undoBtn.addEventListener('click', () => {
      if (this._history.undo()) {
        ui.showNotification('Undo', '');
      } else {
        ui.showNotification('Nothing to undo', 'warning');
      }
    });

    /* Redo */
    ui.redoBtn.addEventListener('click', () => {
      if (this._history.redo()) {
        ui.showNotification('Redo', '');
      } else {
        ui.showNotification('Nothing to redo', 'warning');
      }
    });

    /* Clear */
    ui.clearBtn.addEventListener('click', () => {
      this._clearCanvas();
      ui.showNotification('Canvas cleared', 'success');
    });

    /* Save */
    ui.saveBtn.addEventListener('click', () => {
      this._exportPNG();
    });

    /* Shape toggle */
    ui.shapeToggle.addEventListener('click', () => {
      this._shapeEnabled = !this._shapeEnabled;
      ui.shapeToggle.classList.toggle('active', this._shapeEnabled);
      ui.shapeToggle.setAttribute('aria-pressed', this._shapeEnabled ? 'true' : 'false');
      ui.showNotification(
        this._shapeEnabled ? 'Shape recognition on' : 'Shape recognition off',
        this._shapeEnabled ? 'success' : ''
      );
    });

    /* Debug */
    ui.debugToggleBtn.addEventListener('click', () => {
      this._debugMode = !this._debugMode;
      ui.toggleDebug();
    });

    ui.debugCloseBtn.addEventListener('click', () => {
      this._debugMode = false;
      ui.toggleDebug();
    });

    /* Keyboard shortcuts */
    document.addEventListener('keydown', (e) => {
      if (!this.appScreen?.classList.contains('active') &&
          document.getElementById('app-screen').style.display !== 'block') return;

      const ctrl = e.ctrlKey || e.metaKey;
      switch (e.key) {
        case 'z': case 'Z':
          if (ctrl) {
            e.preventDefault();
            if (e.shiftKey) {
              this._history.redo() ? ui.showNotification('Redo', '') : ui.showNotification('Nothing to redo', 'warning');
            } else {
              this._history.undo() ? ui.showNotification('Undo', '') : ui.showNotification('Nothing to undo', 'warning');
            }
          }
          break;
        case 'y': case 'Y':
          if (ctrl) {
            e.preventDefault();
            this._history.redo() ? ui.showNotification('Redo', '') : ui.showNotification('Nothing to redo', 'warning');
          }
          break;
        case 's': case 'S':
          if (ctrl) { e.preventDefault(); this._exportPNG(); }
          break;
        case 'Delete':
        case 'Backspace':
          if (ctrl) { e.preventDefault(); this._clearCanvas(); ui.showNotification('Canvas cleared', 'success'); }
          break;
        case '1': this._selectToolByKey('pencil'); break;
        case '2': this._selectToolByKey('marker'); break;
        case '3': this._selectToolByKey('neon');   break;
        case '4': this._selectToolByKey('eraser'); break;
        case 'd': case 'D':
          if (!ctrl) { this._debugMode = !this._debugMode; ui.toggleDebug(); }
          break;
      }
    });

    /* Canvas resize → redraw any persistent content */
    window.addEventListener('canvasresize', () => {
      // Camera canvas is auto-redrawn by the render loop.
      // Drawing canvas content is lost on resize — restore from history if possible.
      // (For simplicity, this is a known web canvas limitation.)
    });
  }

  _selectToolByKey(tool) {
    this._brush.tool = tool;
    this._ui.selectTool(tool);
    this._ui.showNotification(`${tool.charAt(0).toUpperCase() + tool.slice(1)} brush`, '');
  }

  /* ── EXPORT ──────────────────────────────────────── */

  _exportPNG() {
    try {
      const dataUrl = this._cm.exportPNG();
      const a = document.createElement('a');
      a.href     = dataUrl;
      a.download = `airdraw-${Date.now()}.png`;
      a.click();
      this._ui.showNotification('Saved as PNG!', 'success');
    } catch (err) {
      console.error('[Export]', err);
      this._ui.showNotification('Export failed', 'error');
    }
  }
}

/* ═══════════════════════════════════════════════════════
   16. BOOTSTRAP
═══════════════════════════════════════════════════════ */
(function bootstrap() {
  'use strict';

  // Guard: ensure we're in a supported browser
  const isSupported =
    typeof navigator.mediaDevices !== 'undefined' &&
    typeof requestAnimationFrame !== 'undefined' &&
    typeof ResizeObserver !== 'undefined';

  if (!isSupported) {
    document.body.innerHTML = `
      <div style="
        display:flex;align-items:center;justify-content:center;
        height:100vh;background:#0a0a0f;color:#fb7185;
        font-family:system-ui;text-align:center;padding:32px;
      ">
        <div>
          <h2 style="font-size:24px;margin-bottom:12px;">Browser Not Supported</h2>
          <p style="color:#64748b;max-width:400px;">
            AirDraw AI Infinity requires a modern browser with camera support.
            Please use Chrome, Edge, or Firefox (latest version).
          </p>
        </div>
      </div>`;
    return;
  }

  // Instantiate and run the app
  const app = new AirDrawApp();

  // Wait for DOM to be fully ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => app.init().catch(console.error));
  } else {
    app.init().catch(console.error);
  }

  // Prevent accidental navigation away
  window.addEventListener('beforeunload', (e) => {
    // Only warn if there's something on canvas — check via history
    // (We can't directly check history without a ref, so we rely on default)
    e.preventDefault();
  });

})();
