// The live waveform: a small pill at the bottom of the screen while the microphone is open.
//
// osascript -l JavaScript level.js <wav>   (spawned by platforms.macos.show_level)
//
// It reads the wav ffmpeg is still writing (16 kHz mono s16le, flushed per packet) and needs no
// IPC: the file growing is "listening", the file no longer growing is "transcribing". The parent
// kills this process once the text has landed. A NON-ACTIVATING panel, because the whole point is
// that the cursor stays where you left it: a window that takes focus takes the paste with it
// (measured: a Tk window does exactly that, this one does not).
ObjC.import("Cocoa");

const BARS = 13;
const BAR_W = 3;
const GAP = 3;
const PAD = 14;
const H = 32;
const MIN_BAR = 3;
const MAX_BAR = 18;
const FPS = 30;
const BYTES_PER_SECOND = 32000;
const HEADER = 44;
// ponytail: fixed ceiling so a killed parent cannot leave a pill on screen forever.
const QUIET_EXIT_SECONDS = 90;

function rgba(r, g, b, a) {
  return $.NSColor.colorWithSRGBRedGreenBlueAlpha(r / 255, g / 255, b / 255, a);
}

// NSBox and not a CALayer: handing a CGColor across the JXA bridge crashes osascript (measured),
// and a custom box takes an NSColor for its fill, edge and radius.
function box(frame, fill, radius) {
  const b = $.NSBox.alloc.initWithFrame(frame);
  b.boxType = 4; // NSBoxCustom
  b.titlePosition = 0; // NSNoTitle
  b.borderWidth = 0;
  b.fillColor = fill;
  b.cornerRadius = radius;
  return b;
}

function run(argv) {
  const wav = argv[0];
  const app = $.NSApplication.sharedApplication;
  app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);

  const width = PAD * 2 + BARS * BAR_W + (BARS - 1) * GAP;
  const screen = $.NSScreen.mainScreen.visibleFrame;
  const x = screen.origin.x + (screen.size.width - width) / 2;
  const y = screen.origin.y + 28;
  // 128 = NSWindowStyleMaskNonactivatingPanel on a borderless (0) panel.
  const panel = $.NSPanel.alloc.initWithContentRectStyleMaskBackingDefer(
    $.NSMakeRect(x, y, width, H), 128, $.NSBackingStoreBuffered, false);
  panel.level = $.NSStatusWindowLevel;
  panel.opaque = false;
  panel.hasShadow = true;
  panel.backgroundColor = $.NSColor.clearColor;
  panel.ignoresMouseEvents = true;
  panel.collectionBehavior = (1 << 0) | (1 << 8); // all Spaces, over full-screen apps

  // The brand's panel ink and hairline edge (docs/index.html: --panel, --edge, --ink).
  const pill = box($.NSMakeRect(0, 0, width, H), rgba(0x16, 0x18, 0x1f, 1), H / 2);
  pill.borderWidth = 1;
  pill.borderColor = rgba(255, 255, 255, 0.09);
  panel.contentView.addSubview(pill);

  const bars = [];
  for (let i = 0; i < BARS; i++) {
    const bar = box($.NSMakeRect(0, 0, BAR_W, MIN_BAR), rgba(0xf6, 0xf5, 0xf1, 1), BAR_W / 2);
    panel.contentView.addSubview(bar);
    bars.push(bar);
  }
  const heights = new Array(BARS).fill(MIN_BAR);

  function draw(opacity) {
    for (let i = 0; i < BARS; i++) {
      const h = heights[i];
      bars[i].frame = $.NSMakeRect(PAD + i * (BAR_W + GAP), (H - h) / 2, BAR_W, h);
      bars[i].alphaValue = opacity[i];
    }
  }

  const fm = $.NSFileManager.defaultManager;
  function size() {
    const attrs = fm.attributesOfItemAtPathError(wav, null);
    // Number(): the bridge hands a long long back as a STRING, and "10968" > "9945" is false.
    return attrs.isNil() ? -1 : Number(attrs.objectForKey($.NSFileSize).longLongValue);
  }

  // One level per frame: the peak of the audio that arrived since the last one, in dBFS, mapped
  // onto a bar. Newest on the right, so speech scrolls in like a waveform.
  let read = HEADER;
  let lastGrowth = Date.now();
  let grew = false;
  function level() {
    const now = size();
    if (now > read + 1) {
      const handle = $.NSFileHandle.fileHandleForReadingAtPath(wav);
      if (!handle.isNil()) {
        const want = Math.min(now - read, BYTES_PER_SECOND / 4) & ~1;
        handle.seekToFileOffset(now - want);
        const data = handle.readDataOfLength(want);
        handle.closeFile;
        read = now;
        lastGrowth = Date.now();
        grew = true;
        const bytes = $.NSString.alloc.initWithDataEncoding(data, $.NSISOLatin1StringEncoding).js;
        let peak = 0;
        for (let i = 0; i + 1 < bytes.length; i += 2) {
          let s = bytes.charCodeAt(i) | (bytes.charCodeAt(i + 1) << 8);
          if (s > 32767) s -= 65536;
          if (Math.abs(s) > peak) peak = Math.abs(s);
        }
        const db = peak > 0 ? 20 * Math.log10(peak / 32768) : -90;
        const t = Math.max(0, Math.min(1, (db + 50) / 40)); // -50 dBFS flat, -10 dBFS full
        return MIN_BAR + t * (MAX_BAR - MIN_BAR);
      }
    }
    return null;
  }

  app.finishLaunching;
  panel.orderFrontRegardless;
  const full = new Array(BARS).fill(1);
  const started = Date.now();
  let frame = 0;
  for (;;) {
    const until = $.NSDate.dateWithTimeIntervalSinceNow(1 / FPS);
    for (;;) {
      const ev = app.nextEventMatchingMaskUntilDateInModeDequeue(-1, until, $.NSDefaultRunLoopMode, true);
      if (ev.isNil()) break;
      app.sendEvent(ev);
    }
    frame++;
    // Read EVERY frame, in both states: growth is the only clock this has. Reading only while
    // listening meant one stalled frame flipped it to "transcribing" for the rest of the clip.
    const h = level();
    const listening = grew ? Date.now() - lastGrowth < 400 : Date.now() - started < 3000;
    if (listening) {
      if (h !== null) {
        heights.shift();
        heights.push(Math.max(h, heights[BARS - 2] * 0.55)); // a short decay, so words read as shapes
      }
      draw(full);
    } else {
      // Transcribing: flat dots and one soft wave of light travelling across them.
      if (Date.now() - lastGrowth > QUIET_EXIT_SECONDS * 1000) return;
      const phase = frame / FPS * 2 * Math.PI * 0.9;
      const glow = [];
      for (let i = 0; i < BARS; i++) {
        heights[i] += (MIN_BAR - heights[i]) * 0.35;
        glow.push(0.25 + 0.75 * Math.pow((Math.sin(phase - i * 0.45) + 1) / 2, 3));
      }
      draw(glow);
    }
  }
}
