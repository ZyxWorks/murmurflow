// THE BACKDROP IS A NIGHT SKY. One module, no dependency, no framework.
//
// It was a particle simulation for one day (2026-08-26) and the operator turned it down the same
// night: "the new backdrop is way too busy... I wanted it to be more like a night sky, sky of
// stars, which move ever so slightly... not have them move with this physics like we have now."
//
// What that measurement actually rejected, so the next pass does not rebuild it:
//
//   * ~50 marks at up to 118px on a 1440px screen. A mark that size is a MARK — it competes with
//     the eight that carry text. A star is 15-29px and reads as distance, not as an object.
//   * elastic collisions. Two ghosts knocking each other sideways is motion with INTENT in it, and
//     the eye follows intent. A sky moves because you are on a planet that turns; nothing in it
//     hits anything.
//   * per-mark rotation. Brand law, and the loudest of the three: `brand/CLAUDE.md` fixes the
//     geometry forever and says never rotated. Fifty tumbling logos is a screensaver.
//
// SO: A JITTERED GRID, THREE SIZES, A SLOW DRIFT AND A TWINKLE. Placement is a grid because the
// alternative is clusters and empty quadrants — the failure mode of every random field, and the
// one thing a real star field never has. Jitter inside the cell is what stops it reading AS a
// grid. Constant area per star, so a phone and a 5K display are the same sky at the same density
// rather than the same COUNT at two densities.
//
// AND ONE LIGHT SOURCE. A sun by day, a moon by night, drawn by the consumer (they are two CSS
// radial gradients, which is the one thing CSS does better than a canvas) and handed to `set()` as
// a position. This file only uses it to LIGHT the field: stars near the body gain a little
// brightness and a little glow, falling off with distance. That is the whole reason the night
// version reads as a sky with a moon in it rather than as dots beside a circle.
//
// STILL ONE MODULE, STILL COPIED OUT. `zyx`'s dashboard, `zyxworks-site` and MurmurFlow's landing
// page all run this file; `make brand-field` overwrites the other two from this one. A React repo
// calls `mount` in a `useEffect`; a plain page calls it in a `<script type="module">`.
//
// CANVAS, NOT DOM NODES. It also makes "never steals a pointer event" true by construction rather
// than by remembering `pointer-events: none`.

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

// THE MARK, AT 120 UNITS, AND IT IS THE IDENTITY'S OWN GEOMETRY.
//
// `brand/CLAUDE.md`: "origin (60,60) -> (60,30) (30,82) (94,74), width 9, round caps... never
// rotated, stretched, filled, gradiented or recolored." The eight marks that carry text already
// draw exactly this; a backdrop drawing a DIFFERENT three-stroke glyph was a second mark on one
// screen, which is the thing brand law is for.
const JOINT = [60, 60];
const TIPS = [
  [60, 30],
  [30, 82],
  [94, 74],
];

/** Three sizes, and the stroke steps with them — the brand's small-mark rule (9 -> 12 -> 14 at
    120 units). At a flat 9 the 15px star is a smudge and the 29px one is a logo. */
export const STARS = [
  { size: 29, stroke: 9 },
  { size: 21, stroke: 12 },
  { size: 15, stroke: 14 },
];

/** One star per this much screen. Density, not count: a phone gets ~8 and a 1440x900 display ~24,
    and both look like the same sky. 230px is the handoff's cell. */
export const CELL = 230;

/** The most stars any screen gets. Above this the cell grows instead — a 5K display at constant
    density would be 150 of them, and the paint is cheap but the SCREEN is not. */
const MAX_STARS = 44;

/** The grid a viewport earns: columns, rows and the cell they sit on. */
export function gridFor(w, h) {
  let cell = CELL;
  let cols = Math.max(1, Math.round(w / cell));
  let rows = Math.max(1, Math.round(h / cell));
  if (cols * rows > MAX_STARS) {
    cell = Math.sqrt((w * h) / MAX_STARS);
    cols = Math.max(1, Math.round(w / cell));
    rows = Math.max(1, Math.round(h / cell));
  }
  return { cols, rows, cell };
}

/** How many stars a viewport earns. Kept as its own export because it is the number anyone
    reviewing "is the backdrop too busy" actually wants to read. */
export function countFor(w, h) {
  const { cols, rows } = gridFor(w, h);
  return cols * rows;
}

/** Night: warm-white. Day: ink. The only two colours in this file — the field is monochrome and
    there is no brass in it. (The handoff asked for one brass star as "a mark that needs a human";
    a backdrop ghost carries no meaning and cannot be pressed, so it would be a FOURTH use of the
    one leashed colour rather than the third. Refused on purpose.) */
export const INK = { dark: "246, 245, 241", light: "11, 12, 16" };

/** Ink on warm-white needs far more alpha than white on near-black to read at all. The handoff
    measured 1.85 and its first draft, which subtracted from the night value instead, made the day
    field vanish completely. */
const DAY_GAIN = 1.85;

/** How far the body's light reaches, as a fraction of the frame's diagonal. */
const HALO_REACH = 0.52;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * PLACE THE FIELD. Pure, seeded, and it never runs again after mount except on a resize.
 *
 * `keepOut` is a list of `{x, y, w, h}` rects in CSS pixels that no star may sit in — the brief's
 * sentence, the command bar, the eight labelled marks. A star at 6% opacity behind a word is not
 * a legibility problem; it is a COMPOSITION one, and the difference between a sky somebody placed
 * and a texture somebody generated. A cell whose jittered point lands in a rect walks out of it
 * rather than being dropped, so the grid keeps its shape and no quadrant goes empty.
 */
export function createField(w, h, { seed = 20260827, keepOut = [] } = {}) {
  const rand = mulberry32(seed);
  const { cols, rows, cell } = gridFor(w, h);
  const cw = w / cols;
  const ch = h / rows;
  const pad = 60; // grown by the keep-out walk below
  const stars = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const kind = STARS[Math.floor(rand() * STARS.length)];
      let x = cw * (c + 0.5) + (rand() - 0.5) * cw * 0.68;
      let y = ch * (r + 0.5) + (rand() - 0.5) * ch * 0.68;
      [x, y] = clear(x, y, kind.size / 2 + 8, keepOut, w, h);
      stars.push({
        x,
        y,
        size: kind.size,
        stroke: kind.stroke,
        // Base opacity, at night. 0.048 … 0.14, and the six brightest are what the moon's glow
        // hangs on — a field where every star is equally bright is a texture.
        a: 0.048 + rand() * 0.092,
        // The drift. Two summed sines per axis at different periods, so the path never repeats
        // visibly and never looks like an orbit. 24-70 second periods: "ever so slightly".
        fx: [0.014 + rand() * 0.014, 0.026 + rand() * 0.016],
        fy: [0.013 + rand() * 0.014, 0.023 + rand() * 0.018],
        px: [rand() * Math.PI * 2, rand() * Math.PI * 2],
        py: [rand() * Math.PI * 2, rand() * Math.PI * 2],
        // The twinkle. A star that only translates reads as a sticker being slid around; the
        // brightness wobble is what makes it a light. Slow, and never below 0.72 of its own base.
        ft: 0.03 + rand() * 0.05,
        pt: rand() * Math.PI * 2,
      });
    }
  }
  // FAR-FIELD DUST, night only. Twelve 2px dots at the back of the room. They are what stops the
  // gaps between stars reading as empty black, and on a light surface they only muddy it — the
  // handoff's own finding, and the reason `paint` skips them by day.
  const dust = [];
  for (let i = 0; i < 12; i += 1) {
    const [dx, dy] = clear(rand() * w, rand() * h, 6, keepOut, w, h);
    dust.push({ x: dx, y: dy, a: 0.08 + rand() * 0.06, ft: 0.02 + rand() * 0.04, pt: rand() * 6.28 });
  }
  return { stars, dust, cell, pad };
}

/** Walk a point out of whatever keep-out rect it landed in, then back inside the frame. A bounded
    push along the shortest axis — four rects and one step each, so there is no loop to run away. */
function clear(x, y, r, keepOut, w, h) {
  for (let pass = 0; pass < 3; pass += 1) {
    let moved = false;
    for (const k of keepOut) {
      const gap = 70; // the handoff's keep-out margin
      const l = k.x - gap - r;
      const t = k.y - gap - r;
      const right = k.x + k.w + gap + r;
      const bottom = k.y + k.h + gap + r;
      if (x <= l || x >= right || y <= t || y >= bottom) continue;
      const out = [x - l, right - x, y - t, bottom - y];
      const min = Math.min(...out);
      if (min === out[0]) x = l;
      else if (min === out[1]) x = right;
      else if (min === out[2]) y = t;
      else y = bottom;
      moved = true;
    }
    if (!moved) break;
  }
  return [clamp(x, r + 4, w - r - 4), clamp(y, r + 4, h - r - 4)];
}

/** Where a star is right now: its anchor plus the drift. Pure, so a check can walk it without a
    canvas — and small on purpose. 7px of wander over half a minute is a sky, 40px is a lava lamp. */
export function driftedAt(star, seconds, amp = 7) {
  return [
    star.x + amp * (Math.sin(seconds * star.fx[0] * 6.283 + star.px[0]) * 0.62 + Math.sin(seconds * star.fx[1] * 6.283 + star.px[1]) * 0.38),
    star.y + amp * (Math.sin(seconds * star.fy[0] * 6.283 + star.py[0]) * 0.62 + Math.sin(seconds * star.fy[1] * 6.283 + star.py[1]) * 0.38),
  ];
}

/** How much of the body's light reaches a point. 1 at the disc, 0 at `HALO_REACH` of the frame's
    diagonal, linear in between — the handoff's own falloff. */
function lightAt(x, y, body, w, h) {
  if (!body) return 0;
  const reach = Math.hypot(w, h) * HALO_REACH;
  return clamp(1 - Math.hypot(x - body.x, y - body.y) / reach, 0, 1);
}

/**
 * DRAW ONE FRAME.
 *
 *   `theme` "dark" | "light"     which ink, and whether there is glow and dust at all
 *   `body`  {x, y} | null        where the sun or moon is, in CSS pixels
 *   `dusk`  0…1                  how far into civil twilight; fades the night field up
 *
 * Day and night are not the same picture with a colour swapped. At night the moon LIGHTS the
 * field — every star carries a glow, and the ones near the disc carry more. By day the sun DIMS
 * it: no glow at all (a white wash around a dark mark on a warm-white page reads as a printing
 * fault), a much higher base alpha, and stars inside the sun's wash lose a little rather than
 * being erased.
 */
export function paint(ctx, field, w, h, { seconds = 0, theme = "dark", body = null, dusk = 1 } = {}) {
  const night = theme !== "light";
  const rgb = night ? INK.dark : INK.light;
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";

  if (night) {
    for (const d of field.dust) {
      const tw = 0.82 + 0.18 * Math.sin(seconds * d.ft * 6.283 + d.pt);
      ctx.fillStyle = `rgba(${rgb}, ${(d.a * tw * dusk).toFixed(4)})`;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 1, 0, 6.2832);
      ctx.fill();
    }
  }

  for (const s of field.stars) {
    const [x, y] = driftedAt(s, seconds);
    const lit = lightAt(x, y, body, w, h);
    const twinkle = 0.86 + 0.14 * Math.sin(seconds * s.ft * 6.283 + s.pt);
    let alpha;
    if (night) {
      alpha = (s.a + 0.035 * lit) * twinkle * dusk;
    } else {
      // Floor 0.09: the first draft subtracted a flat 0.05 off a 0.04 base and the whole day field
      // disappeared. A mark must stay readable everywhere on a light surface.
      alpha = Math.max(0.09, s.a * DAY_GAIN - 0.02 * lit) * twinkle;
    }

    if (night) {
      // The wash. `radial-gradient(circle, currentColor 0%, transparent 52%)` blurred 6px, drawn
      // as a gradient with a soft shoulder instead — a blur filter on a canvas is a per-frame
      // readback on some drivers, and a gradient is already the shape a blur was there to make.
      // Never above 0.11: at the ring it stops being light and becomes an outline.
      const g = Math.min(0.075, (0.018 + (s.a - 0.048) * 0.34 + 0.022 * lit) * dusk);
      const rad = s.size * 0.95;
      const grad = ctx.createRadialGradient(x, y, 0, x, y, rad);
      grad.addColorStop(0, `rgba(${rgb}, ${g.toFixed(4)})`);
      // The falloff is a SQUARE, not a shoulder. A stop partway out is a second edge, and the
      // first version of this drew a visible 44px disc around every star - the exact "reads as a
      // ring around each mark" the handoff warned about, arrived at from the other direction.
      grad.addColorStop(0.45, `rgba(${rgb}, ${(g * 0.3).toFixed(4)})`);
      grad.addColorStop(0.75, `rgba(${rgb}, ${(g * 0.07).toFixed(4)})`);
      grad.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, 6.2832);
      ctx.fill();
    }

    const k = s.size / 120;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(k, k);
    ctx.translate(-JOINT[0], -JOINT[1]);
    ctx.strokeStyle = `rgba(${rgb}, ${alpha.toFixed(4)})`;
    ctx.lineWidth = s.stroke;
    ctx.beginPath();
    for (const [tx, ty] of TIPS) {
      ctx.moveTo(JOINT[0], JOINT[1]);
      ctx.lineTo(tx, ty);
    }
    ctx.stroke();
    ctx.restore();
  }
}

/**
 * Mount the field on a canvas. Returns `{ stop, set }`.
 *
 * `set({theme, body, dusk, keepOut})` is how the consumer moves the light: the sky's own state
 * changes about once a minute, and a repaint on demand is cheaper and simpler than handing this
 * module a clock and a location. Under reduced motion it repaints the one static frame.
 *
 * THE BUDGET, because it runs forever behind everything. Count is capped by AREA and by
 * `MAX_STARS`; DPR is capped at 2; `visibilitychange` stops the loop outright so a background tab
 * costs zero; an `IntersectionObserver` does the same where the canvas can scroll out of view,
 * which is the marketing site and not the dashboard. There is no broadphase and no pair loop any
 * more — a drift is `sin` twice per axis, so the whole frame is ~24 stars of trigonometry.
 *
 * `prefers-reduced-motion` renders ONE STATIC FRAME and never starts a loop. Not a slowed
 * simulation — a still picture. Accessibility basic, and non-negotiable.
 */
export function mount(canvas, { seed = 20260827, reducedMotion, observe = false, ...initial } = {}) {
  if (!canvas?.getContext) return { stop() {}, set() {} };
  const ctx = canvas.getContext("2d");
  const still =
    reducedMotion ??
    (typeof window !== "undefined" &&
      !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);

  let state = { theme: "dark", body: null, dusk: 1, keepOut: [], ...initial };
  let field = null;
  let w = 0;
  let h = 0;
  let raf = 0;
  let t0 = 0;
  let onScreen = true;
  let stopped = false;
  let watcher = null;

  function size() {
    const box = canvas.getBoundingClientRect();
    const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
    w = Math.max(1, Math.round(box.width || canvas.width || 1));
    h = Math.max(1, Math.round(box.height || canvas.height || 1));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    field = createField(w, h, { seed, keepOut: state.keepOut });
  }

  // Under reduced motion `seconds` is pinned to 0, so the still frame is the field's own anchors
  // with no drift and no twinkle applied — the picture it would settle to, not a random instant.
  const draw = () =>
    paint(ctx, field, w, h, {
      seconds: still || !t0 ? 0 : (performance.now() - t0) / 1000,
      ...state,
    });

  function frame() {
    raf = 0;
    if (stopped) return;
    draw();
    schedule();
  }

  function schedule() {
    if (stopped || still || raf || document.hidden || !onScreen) return;
    raf = requestAnimationFrame(frame);
  }

  const onResize = () => {
    size();
    draw();
  };
  const onVisible = () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
      raf = 0;
    } else schedule();
  };

  size();
  t0 = typeof performance !== "undefined" ? performance.now() : 0;
  draw();

  const api = {
    /** Move the light, flip the theme, or hand over new keep-out rects. Only a keep-out change
        re-places the field: everything else is a repaint, so the sky does not reshuffle at dawn. */
    set(next = {}) {
      const rebuild = "keepOut" in next && next.keepOut !== state.keepOut;
      state = { ...state, ...next };
      if (rebuild) field = createField(w, h, { seed, keepOut: state.keepOut });
      if (still || rebuild) draw();
    },
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      if (typeof window === "undefined") return;
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisible);
      watcher?.disconnect();
    },
  };

  if (still) return api; // one frame, and no loop — but `set` still repaints on a theme flip

  window.addEventListener("resize", onResize);
  document.addEventListener("visibilitychange", onVisible);
  if (observe && typeof IntersectionObserver !== "undefined") {
    watcher = new IntersectionObserver((rows) => {
      onScreen = rows.some((r) => r.isIntersecting);
      if (onScreen) schedule();
    });
    watcher.observe(canvas);
  }
  schedule();
  return api;
}
