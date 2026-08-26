// THE BACKDROP IS A REAL PARTICLE FIELD. One module, no dependency, no framework.
//
// Part two of the sky's physics, and deliberately a SEPARATE feature from part one. The eight
// marks carry TEXT, so they get leashes and a proof that no pair can ever touch (`geometry.js`).
// A ghost back here carries nothing, is `aria-hidden`, sits behind everything and is read by
// nobody — so it is the one layer where a collision costs zero, and therefore the one layer where
// bouncing is free. Nothing in this file may ever reach into that one.
//
// WHY IT IS NOT A REACT COMPONENT, AND NOT IN `ui-kit`. MurmurFlow's landing page is plain HTML
// with no React at all, and a hook would force a framework on every consumer for a decorative
// canvas. `@products/ui-kit` exists, has zero deps and React as a peer, and has no consumers —
// making it the home means building a package chain before writing a line of physics. So: one
// vanilla ES module. A React repo calls `mount` in a `useEffect`; a plain page calls it in a
// `<script type="module">`.
//
// IT IS ALREADY DUPLICATED, AND THE COPIES HAVE DIVERGED. That is the argument for this file, and
// it is a measurement rather than a preference: `zyx`'s Sky.jsx and `zyxworks-site`'s Backdrop.jsx
// hand-place nine marks each at DIFFERENT positions, share six drift frequencies by hand-copy, and
// one of them seeds its phases from `Math.random()` — so the marketing site reshuffles its sky on
// every reload, which the dashboard explicitly refuses to do. MurmurFlow's page is a third,
// hand-ported copy of the same idea in vanilla JS.
//
// CANVAS, NOT NINE DOM NODES. Nine `<span>`s are fine for nine ghosts. A field wants 30-60, each
// with position, velocity, depth and a collision pass — and a canvas also makes "never steals a
// pointer event" true by construction rather than by remembering `pointer-events: none`.

/** A 5-line PRNG. NEVER `Math.random`: a reload must not reshuffle the sky, and the marketing
    site's version doing exactly that is the bug this file exists not to carry over. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The depth ramp, near to far. Verbatim the tones the dashboard's own ghosts already use. */
export const TONES = ["#3a3d47", "#343741", "#2a2d36", "#22242c", "#1f2127"];

/** How many particles a viewport earns. Not a constant: this runs FOREVER behind everything, so
    the budget is a property of the screen it is drawn on. ~12 on a 390px phone, ~50 at 1440x900. */
export function countFor(w, h) {
  return Math.max(6, Math.min(60, Math.floor((w * h) / 26000)));
}

const MIN_Z = 0.06; // never all the way to nothing: a particle at z=0 is a particle that vanished
const BAND = 0.22; // two particles collide only if their depths are this close

/**
 * THE NEAREST PARTICLE'S SIZE IS A FRACTION OF THE SCREEN, not a constant.
 *
 * 118px is right on a desktop and is a third of a 390px phone — and MurmurFlow's own hand-ported
 * copy of this field had already learned that the hard way, carrying a whole SECOND array of
 * "fewer, smaller marks pinned to the margins" below 900px, because "the desktop placement puts a
 * 116px mark straight through a headline". A second hand-placed field is exactly what this module
 * exists to delete, so the lesson comes with it rather than the array: size follows the smaller
 * viewport dimension, and every consumer gets the phone case right without knowing about it.
 *
 * 64 at 390x844, 117 at 1440x900, 118 (the ceiling) on anything wider.
 */
export const nearSize = (w, h) => Math.max(64, Math.min(118, Math.min(w, h) * 0.13));
const FAR_RATIO = 0.22; // the farthest is about a fifth of the nearest, at any size

/** A particle's drawn size, from its depth. The one place size and depth are tied together.
    Squared, so it reads as perspective rather than as a straight ramp. */
export const sizeAt = (z, near = 118) => near * (FAR_RATIO + (1 - FAR_RATIO) * z * z);

/**
 * THE SIMULATION, WITHOUT A CANVAS. Exported on its own so the physics can be checked by a script
 * with no browser in it — which is the whole reason the drift solve in `geometry.js` is pure too.
 *
 * DEPTH IS REAL, AND DEPTH MOVES (operator, 2026-08-26: "make the marks in the backdrop grow and
 * go smaller so it seems like they go far into the backdrop or come near again... a little bit
 * like small shooting stars"). `z` drives size, tone AND speed together — far ones smaller, dimmer
 * and slower — and it has a velocity of its own, so a particle genuinely travels toward you and
 * away again. The plan had `z` as a fixed per-particle constant; a static ramp reads as a flat
 * sheet of dots at three greys, which is exactly what the field looked like before.
 *
 * COLLISIONS ONLY WITHIN A DEPTH BAND. Cheap, and it is also what a real depth field looks like:
 * things far away do not hit things near you. Elastic, equal mass, no rotation transfer.
 *
 * A UNIFORM GRID, not a pair loop. 60 particles is 1770 pair checks a frame, forever, behind
 * everything — and the grid makes it the handful of neighbours each particle actually has.
 */
export function createField(w, h, { count = countFor(w, h), seed = 20260826 } = {}) {
  const rand = mulberry32(seed);
  const near = nearSize(w, h);
  const parts = [];
  for (let i = 0; i < count; i += 1) {
    const z = MIN_Z + rand() * (1 - MIN_Z);
    parts.push({
      x: rand() * w,
      y: rand() * h,
      z,
      // Speed scales with depth: a near particle crosses the screen, a far one barely moves. That
      // is what makes the field read as distance rather than as a sheet of dots at three greys.
      vx: (rand() - 0.5) * 14 * (0.25 + z),
      vy: (rand() - 0.5) * 14 * (0.25 + z),
      vz: (rand() - 0.5) * 0.035,
      spin: rand() * Math.PI * 2,
      spinRate: (rand() - 0.5) * 0.22,
    });
  }

  /** One step. `dt` in seconds, clamped by the caller — a backgrounded tab hands back a huge one. */
  function step(dt, width = w, height = h) {
    for (const p of parts) {
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
      p.spin += p.spinRate * dt;
      // The depth band is a wall like any other, so a particle coming toward you turns around and
      // goes back rather than clipping to the front and sticking there.
      if (p.z < MIN_Z) {
        p.z = MIN_Z;
        p.vz = Math.abs(p.vz);
      } else if (p.z > 1) {
        p.z = 1;
        p.vz = -Math.abs(p.vz);
      }
      const r = sizeAt(p.z, near) / 2;
      if (p.x < r) {
        p.x = r;
        p.vx = Math.abs(p.vx);
      } else if (p.x > width - r) {
        p.x = width - r;
        p.vx = -Math.abs(p.vx);
      }
      if (p.y < r) {
        p.y = r;
        p.vy = Math.abs(p.vy);
      } else if (p.y > height - r) {
        p.y = height - r;
        p.vy = -Math.abs(p.vy);
      }
    }

    // --- broadphase: one uniform grid, cell = the largest particle ---------------------------
    const cell = near;
    const cols = Math.max(1, Math.ceil(width / cell));
    const buckets = new Map();
    for (let i = 0; i < parts.length; i += 1) {
      const p = parts[i];
      const key = Math.floor(p.y / cell) * cols + Math.floor(p.x / cell);
      const at = buckets.get(key);
      if (at) at.push(i);
      else buckets.set(key, [i]);
    }
    const around = [];
    for (const [key, ids] of buckets) {
      const cx = key % cols;
      const cy = (key - cx) / cols;
      around.length = 0;
      // Only forward neighbours, so every pair is visited exactly once.
      for (const [dx, dy] of [[0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]) {
        const other = buckets.get((cy + dy) * cols + (cx + dx));
        if (other && !(dx === 0 && dy === 0)) around.push(...other);
      }
      for (let a = 0; a < ids.length; a += 1) {
        for (let b = a + 1; b < ids.length; b += 1) hit(parts[ids[a]], parts[ids[b]]);
        for (const id of around) hit(parts[ids[a]], parts[id]);
      }
    }
    return parts;
  }

  function hit(p, q) {
    if (Math.abs(p.z - q.z) > BAND) return; // different distances: it passes behind
    const rp = sizeAt(p.z, near) / 2;
    const rq = sizeAt(q.z, near) / 2;
    const dx = q.x - p.x;
    const dy = q.y - p.y;
    const d2 = dx * dx + dy * dy;
    const reach = rp + rq;
    if (d2 >= reach * reach || d2 === 0) return;
    const d = Math.sqrt(d2);
    const nx = dx / d;
    const ny = dy / d;
    // Equal mass, elastic: they swap the component of their velocity along the line between them.
    const along = (p.vx - q.vx) * nx + (p.vy - q.vy) * ny;
    if (along > 0) {
      p.vx -= along * nx;
      p.vy -= along * ny;
      q.vx += along * nx;
      q.vy += along * ny;
    }
    // ...and are pushed apart, so a pair that arrives overlapping cannot stay stuck together.
    const push = (reach - d) / 2;
    p.x -= nx * push;
    p.y -= ny * push;
    q.x += nx * push;
    q.y += ny * push;
  }

  return { parts, step, near };
}

/**
 * Draw one frame. The mark is the UPRIGHT variant — three round-capped strokes from one joint,
 * symmetric — and that is brand law rather than taste: upright means "a place you can go", the
 * asymmetric one means zyx itself, and there is exactly ONE asymmetric mark on a screen (the
 * wordmark). `sky-physics-plan.md` quotes the identity's own stroke list here, which would have
 * put fifty logos in the background.
 */
export function paint(ctx, parts, w, h, near = nearSize(w, h)) {
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";
  // Far first, so a near particle genuinely passes in front of a far one.
  for (const p of [...parts].sort((a, b) => a.z - b.z)) {
    const s = sizeAt(p.z, near) / 120;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.spin);
    ctx.scale(s, s);
    ctx.translate(-60, -62);
    ctx.strokeStyle = TONES[Math.min(TONES.length - 1, Math.floor((1 - p.z) * TONES.length))];
    ctx.lineWidth = 9;
    ctx.beginPath();
    ctx.moveTo(60, 30);
    ctx.lineTo(60, 62);
    ctx.moveTo(32, 82);
    ctx.lineTo(60, 62);
    ctx.moveTo(88, 82);
    ctx.lineTo(60, 62);
    ctx.stroke();
    ctx.restore();
  }
}

/**
 * Mount the field on a canvas. Returns `{ stop }`.
 *
 * THE BUDGET, because it runs forever behind everything. Every line of it is code rather than a
 * hope: the count is capped by viewport AREA and not by a constant; DPR is capped at 2 (a 3x canvas
 * is 2.25x the pixels for nothing anyone can see on a backdrop); `visibilitychange` stops the loop
 * outright so a background tab costs zero; and an `IntersectionObserver` does the same where the
 * canvas can scroll out of view, which is the marketing site and not the dashboard.
 *
 * `prefers-reduced-motion` renders ONE STATIC FRAME and never starts a loop. Not a slowed
 * simulation — a still picture. Accessibility basic, and non-negotiable.
 *
 * MEASURED, because "it should not take any CPU" is a claim and not a hope (operator, 2026-08-26).
 * The physics, timed over 6000 frames after a warm-up:
 *
 *     1440x900   49 particles   0.042 ms/frame   0.25% of a 60fps budget
 *     390x844    12 particles   0.010 ms/frame   0.06%
 *     2560x1440  60 particles   0.027 ms/frame   0.16%
 *
 * The 2560 case is FASTER than the 1440 one per frame because the count is capped at 60 while the
 * grid gets bigger — fewer neighbours per cell. That is the broadphase doing its job; a pair loop
 * would have gone the other way. Driven in a real browser as well: a hidden tab is bit-identical
 * after 2.5 seconds, and a reduced-motion context paints once and never again.
 */
export function mount(canvas, { seed = 20260826, count, reducedMotion, observe = false } = {}) {
  if (!canvas?.getContext) return { stop() {} };
  const ctx = canvas.getContext("2d");
  const still =
    reducedMotion ??
    (typeof window !== "undefined" &&
      !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

  let field = null;
  let w = 0;
  let h = 0;
  let raf = 0;
  let last = 0;
  let onScreen = true;
  let stopped = false;

  function size() {
    const box = canvas.getBoundingClientRect();
    const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
    w = Math.max(1, Math.round(box.width || canvas.width || 1));
    h = Math.max(1, Math.round(box.height || canvas.height || 1));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    field = createField(w, h, { count, seed });
  }

  function frame(now) {
    raf = 0;
    if (stopped) return;
    // Clamped: a tab that was hidden hands back a dt of minutes, and the first step after would
    // teleport every particle across the screen and through every wall test.
    const dt = Math.min(0.05, last ? (now - last) / 1000 : 0.016);
    last = now;
    field.step(dt, w, h);
    paint(ctx, field.parts, w, h, field.near);
    schedule();
  }

  function schedule() {
    if (stopped || still || raf || document.hidden || !onScreen) return;
    raf = requestAnimationFrame(frame);
  }

  function wake() {
    last = 0; // the clock restarts with the loop; see the dt clamp above
    schedule();
  }

  const onResize = () => {
    size();
    paint(ctx, field.parts, w, h, field.near);
  };
  const onVisible = () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
      raf = 0;
    } else wake();
  };

  size();
  paint(ctx, field.parts, w, h, field.near);
  if (still) return { stop() {} }; // one frame, and no listeners to leak

  window.addEventListener("resize", onResize);
  document.addEventListener("visibilitychange", onVisible);
  let watcher = null;
  if (observe && typeof IntersectionObserver !== "undefined") {
    watcher = new IntersectionObserver((rows) => {
      onScreen = rows.some((r) => r.isIntersecting);
      if (onScreen) wake();
    });
    watcher.observe(canvas);
  }
  schedule();

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisible);
      watcher?.disconnect();
    },
  };
}
