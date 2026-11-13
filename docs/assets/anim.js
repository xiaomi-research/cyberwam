/* Page animations: scroll reveal, number count-up, and the 3-D timestep-space widget. No dependencies. */
(function () {
  "use strict";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.documentElement.classList.add("js");

  /* ---------- scroll reveal ---------- */
  var revealEls = document.querySelectorAll(".reveal, .sec-head, .stat > div");
  revealEls.forEach(function (el) { el.classList.add("reveal"); });
  if ("IntersectionObserver" in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- count-up ---------- */
  function countUp(el) {
    var target = parseFloat(el.getAttribute("data-count")), dec = +(el.getAttribute("data-dec") || 0);
    var pre = el.getAttribute("data-prefix") || "", suf = el.getAttribute("data-suffix") || "";
    if (reduce) { el.textContent = pre + target.toFixed(dec) + suf; return; }
    var t0 = null, dur = 1200;
    function step(ts) {
      if (t0 === null) t0 = ts;
      var k = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = pre + (target * e).toFixed(dec) + suf;
      if (k < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var counters = document.querySelectorAll("[data-count]");
  if ("IntersectionObserver" in window) {
    var io2 = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { countUp(e.target); io2.unobserve(e.target); } });
    }, { threshold: 0.6 });
    counters.forEach(function (el) { io2.observe(el); });
  } else { counters.forEach(countUp); }

  /* ---------- reading progress + active nav link ---------- */
  var prog = document.getElementById("prog");
  function onScroll() {
    var h = document.documentElement.scrollHeight - window.innerHeight;
    if (prog && h > 0) prog.style.width = Math.min(100, (window.pageYOffset / h) * 100) + "%";
  }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  var links = document.querySelectorAll("nav.toc a[href^='#']");
  if ("IntersectionObserver" in window && links.length) {
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (a) { a.classList.remove("act"); });
        var a = map[e.target.id]; if (a) { a.classList.add("act"); }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(map).forEach(function (id) { var el = document.getElementById(id); if (el) spy.observe(el); });
  }

  /* ---------- 3-D timestep space ---------- */
  var canvas = document.getElementById("tsw-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var OR = [242, 107, 29], GR = [189, 181, 173], INK = "#25302E", MUT = "#7D8A86";
  var D = 0.1, N = 600;

  // deterministic pseudo-random numbers so that the figure looks the same on every load
  var seed = 12345;
  function rnd() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
  var P = [];                       // independent samples
  for (var i = 0; i < N; i++) {
    var p = [rnd(), rnd(), rnd()];
    var mx = Math.max(p[0], p[1], p[2]), mn = Math.min(p[0], p[1], p[2]);
    var m = (p[0] + p[1] + p[2]) / 3;
    P.push({ p: p, m: m, near: (mx - mn) < D });
  }
  var nNear = P.filter(function (q) { return q.near; }).length;

  var yaw = 0.75, pitch = 0.32, auto = !reduce, dragging = false, lastX = 0, lastY = 0;
  var mode = 0, modeT = 0;          // mode 0 = independent, 1 = shared; modeT animates 0..1
  var tracer = 0, running = true, visible = true;
  var W = 0, H = 0, dpr = 1;

  function resize() {
    var r = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);

  function proj(x, y, z) {
    x -= 0.5; y -= 0.5; z -= 0.5;
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    var y1 = cp * y - sp * z1, z2 = sp * y + cp * z1;
    var S = Math.min(W * 0.42, H * 0.62);
    var f = 3.2 / (3.2 - z2);
    return [W * 0.5 + x1 * S * f, H * 0.5 - y1 * S * f, z2, f];
  }
  function mix(a, b, k) { return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]; }
  function rgba(c, a) { return "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + a + ")"; }
  function ease(k) { return k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2; }

  var EDGES = [[0,0,0,1,0,0],[0,1,0,1,1,0],[0,0,1,1,0,1],[0,1,1,1,1,1],
               [0,0,0,0,1,0],[1,0,0,1,1,0],[0,0,1,0,1,1],[1,0,1,1,1,1],
               [0,0,0,0,0,1],[1,0,0,1,0,1],[0,1,0,0,1,1],[1,1,0,1,1,1]];

  function line(a, b, style, w, dash) {
    var pa = proj(a[0], a[1], a[2]), pb = proj(b[0], b[1], b[2]);
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
    ctx.strokeStyle = style; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
  }
  function label(txt, x, y, z, color, align, bold) {
    var p = proj(x, y, z);
    ctx.font = (bold ? "700 " : "500 ") + (W < 480 ? 10.5 : 12.5) + "px Helvetica, Arial, sans-serif";
    ctx.fillStyle = color; ctx.textAlign = align || "center"; ctx.textBaseline = "middle";
    ctx.fillText(txt, p[0], p[1]);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // cube
    EDGES.forEach(function (e) { line([e[0], e[1], e[2]], [e[3], e[4], e[5]], "rgba(125,138,134,0.35)", 1); });
    // diagonal band (the inference path), drawn wide and faint, then a dashed core
    line([0, 0, 0], [1, 1, 1], "rgba(242,107,29,0.14)", 16);
    line([0, 0, 0], [1, 1, 1], "rgba(242,107,29,0.85)", 1.6, [5, 4]);

    // points, back to front
    var k = ease(modeT), items = [];
    for (var i = 0; i < N; i++) {
      var q = P[i];
      var x = q.p[0] + (q.m - q.p[0]) * k, y = q.p[1] + (q.m - q.p[1]) * k, z = q.p[2] + (q.m - q.p[2]) * k;
      var pr = proj(x, y, z);
      items.push([pr[2], pr[0], pr[1], pr[3], q.near]);
    }
    items.sort(function (a, b) { return a[0] - b[0]; });
    for (var j = 0; j < items.length; j++) {
      var it = items[j], near = it[4];
      var col = near ? OR : mix(GR, OR, k);
      var r = (near ? 3.1 : 2.3) * it[3] * (W < 480 ? 0.85 : 1);
      ctx.beginPath(); ctx.arc(it[1], it[2], r, 0, 6.2832);
      ctx.fillStyle = rgba(col, near ? 0.95 : (0.42 + 0.5 * k)); ctx.fill();
    }

    // inference tracer: noise (1,1,1) -> clean (0,0,0)
    var u = 1 - tracer;
    var tp = proj(u, u, u);
    for (var s = 1; s <= 10; s++) {
      var uu = Math.min(1, u + s * 0.018), tq = proj(uu, uu, uu);
      ctx.beginPath(); ctx.arc(tq[0], tq[1], (6 - s * 0.45) * tq[3], 0, 6.2832);
      ctx.fillStyle = "rgba(232,89,12," + (0.22 - s * 0.02) + ")"; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(tp[0], tp[1], 7.5 * tp[3], 0, 6.2832);
    ctx.fillStyle = "#E8590C"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();

    // labels
    label("RGB timestep", 1.0, -0.09, 0.0, INK, "center", true);
    label("DINO timestep", -0.04, -0.09, 1.0, INK, "center", true);
    label("action timestep", -0.09, 1.07, 0.0, INK, "center", true);
    label("pure noise", 1.0, 1.0, 1.0, "#D9500A", "left", true);
    label("clean", 0.0, 0.0, 0.0, MUT, "right", false);
  }

  var last = 0;
  function frame(ts) {
    if (!running) return;
    var dt = Math.min(50, ts - last); last = ts;
    if (visible) {
      if (auto && !dragging) yaw += dt * 0.00022;
      var goal = mode; modeT += (goal - modeT) * Math.min(1, dt * 0.006);
      if (Math.abs(goal - modeT) < 0.002) modeT = goal;
      if (!reduce) { tracer += dt / 3600; if (tracer > 1.25) tracer = 0; if (tracer > 1) tracer = 1; }
      else { tracer = 0.55; }
      draw();
    }
    requestAnimationFrame(frame);
  }

  /* interaction */
  canvas.addEventListener("pointerdown", function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    yaw += (e.clientX - lastX) * 0.008;
    pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - lastY) * 0.006));
    lastX = e.clientX; lastY = e.clientY;
  });
  function up() { dragging = false; }
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);

  var btnI = document.getElementById("tsw-ind"), btnS = document.getElementById("tsw-sh");
  var stat = document.getElementById("tsw-stat"), auto_cb = document.getElementById("tsw-auto");
  function setMode(m) {
    mode = m;
    btnI.classList.toggle("on", m === 0); btnS.classList.toggle("on", m === 1);
    btnI.setAttribute("aria-pressed", m === 0); btnS.setAttribute("aria-pressed", m === 1);
    stat.innerHTML = m === 0
      ? "<b>" + nNear + "</b> of " + N + " samples lie near the inference path <span>(about 2.8% in expectation)</span>"
      : "<b>" + N + "</b> of " + N + " samples lie on the inference path";
  }
  btnI.addEventListener("click", function () { setMode(0); });
  btnS.addEventListener("click", function () { setMode(1); });
  auto_cb.checked = auto;
  auto_cb.addEventListener("change", function () { auto = auto_cb.checked; });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (en) { visible = en[0].isIntersecting; }, { threshold: 0.02 }).observe(canvas);
  }
  resize(); setMode(0); modeT = 0;
  requestAnimationFrame(function (ts) { last = ts; frame(ts); });
})();
