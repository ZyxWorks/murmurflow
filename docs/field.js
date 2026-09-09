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
//   * per-mark rotation IN THE PLANE. Fifty logos spinning like a loading spinner is a
//     screensaver, and `brand/CLAUDE.md` fixes the geometry forever. A TURN IN 3D IS NOT THAT AND
//     IS NOW WHAT THIS FILE DOES - see AXES below: the mark is an xyz axis, so the object is fixed
//     and only the angle you see it from moves. The geometry is never re-drawn.
//
// SO: A JITTERED GRID, THREE SIZES, A SLOW DRIFT, A TWINKLE AND A SLOW TURN IN 3D. Placement is a grid because the
// alternative is clusters and empty quadrants — the failure mode of every random field, and the
// one thing a real star field never has. Jitter inside the cell is what stops it reading AS a
// grid. Constant area per star, so a phone and a 5K display are the same sky at the same density
// rather than the same COUNT at two densities.
//
// A STAR IS A CRISP GLYPH AND NOTHING ELSE (SKY-QUIET-1, operator, 2026-08-27).
//
// It used to carry a radial wash behind it, and to gain brightness and glow from the body passing
// nearby. Both are deleted. Over a real screen the wash read as a smudge behind every mark and the
// body's light read as a lamp somebody had left on: "the light from the stars is a little too much
// for some reason... it looks a little too kitschy". A field of hairline glyphs at 5-14% is the
// same language as the eight marks that carry text, which is the point.
//
// So this file no longer knows where the sun or the moon is at all. `paint` still ACCEPTS a `body`
// so the two copied-out consumers do not break on an extra key, and does nothing with it.
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

// AND THE MARK IS AN XYZ AXIS, SO IT TURNS ON ONE (operator, 2026-08-27: "I want this xyz thingy
// to rotate in the xyz axis, right? Cause Zyx is an xyz axis").
//
// These are the brand's own three strokes lifted back into 3D. Drop the z and you get
// (60,30) (30,82) (94,74) EXACTLY - so nothing is re-drawn and nothing is a new glyph. The z each
// stroke carries is the one that makes the three mutually PERPENDICULAR, which is what makes the
// mark a real axis triad rather than three lines that happen to meet: solved once from
// `v1.v2 + z1*z2 = 0` and its two siblings (z1*z2 = 660, z1*z3 = 420, z2*z3 = 712), and pinned by
// `check-registry.mjs` so nobody has to re-derive it.
//
// The turn is a SWAY of about 11 degrees, never a spin. An axis rotated far enough to point at the
// viewer foreshortens to a dot, and a field of marks blinking out is the screensaver again.
const AXES = [
  [0, -30, 19.731],
  [-30, 22, 33.449],
  [34, 14, 21.286],
];

/** How far a star's mark may turn, in radians: yaw about the vertical, pitch about the horizontal. */
export const TURN = [0.3, 0.18];

/** The three tips of the mark as seen from `yaw`/`pitch`, projected by dropping z. At (0, 0) this
    returns the brand geometry to the pixel. Pure, so a check can walk it without a canvas. */
export function tipsAt(yaw, pitch) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  return AXES.map(([x, y, z]) => {
    const rx = x * cy + z * sy;
    const rz = z * cy - x * sy;
    return [JOINT[0] + rx, JOINT[1] + y * cp - rz * sp];
  });
}

/** Where a star's mark is pointing right now. Same two-summed-sines shape as the drift, so the
    angle never repeats visibly and never looks like a motor. */
export function turnedAt(star, seconds) {
  return [
    TURN[0] * Math.sin(seconds * star.fr * 6.283 + star.pr),
    TURN[1] * Math.sin(seconds * star.fq * 6.283 + star.pq),
  ];
}

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
        // The turn. 25-90 second periods, deliberately slower than the twinkle and slower than the
        // drift: the nearest thing to it in the real world is a planet turning, not a mobile.
        fr: 0.011 + rand() * 0.029,
        pr: rand() * Math.PI * 2,
        fq: 0.011 + rand() * 0.022,
        pq: rand() * Math.PI * 2,
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

/**
 * DRAW ONE FRAME.
 *
 *   `theme` "dark" | "light"     which ink, and whether there is dust at all
 *   `dusk`  0…1                  how far into civil twilight; fades the night field up
 *
 * Day and night are the same picture in two inks and two alphas. Ink on warm-white needs far more
 * of it to read at all (`DAY_GAIN`), and the far-field dust is night-only because on a light
 * surface it only muddies it. There is no glow in either — see the head of this file.
 */
export function paint(ctx, field, w, h, { seconds = 0, theme = "dark", dusk = 1 } = {}) {
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
    const twinkle = 0.86 + 0.14 * Math.sin(seconds * s.ft * 6.283 + s.pt);
    // Floor 0.09 by day: an early draft let a mark fall under it and the whole day field
    // disappeared. A mark must stay readable everywhere on a light surface.
    const alpha = night ? s.a * twinkle * dusk : Math.max(0.09, s.a * DAY_GAIN) * twinkle;

    const k = s.size / 120;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(k, k);
    ctx.translate(-JOINT[0], -JOINT[1]);
    ctx.strokeStyle = `rgba(${rgb}, ${alpha.toFixed(4)})`;
    ctx.lineWidth = s.stroke;
    ctx.beginPath();
    for (const [tx, ty] of tipsAt(...turnedAt(s, seconds))) {
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
 * `set({theme, dusk, keepOut})` is how the consumer turns the sky over: its state changes about
 * once a minute, and a repaint on demand is cheaper and simpler than handing this module a clock
 * and a location. Under reduced motion it repaints the one static frame.
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

  let state = { theme: "dark", dusk: 1, keepOut: [], ...initial };
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
    /** Flip the theme, move through twilight, or hand over new keep-out rects. Only a keep-out
        change re-places the field: everything else is a repaint, so it does not reshuffle at dawn. */
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
