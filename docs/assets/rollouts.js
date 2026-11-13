/* Rollout player: plays a time-lapse video of a successful episode of the final model. */
(function () {
  "use strict";
  var TASKS = [{"id": 24, "name": "Storing Food", "total": 601}, {"id": 21, "name": "Collecting Children's Toys", "total": 467}, {"id": 27, "name": "Sorting Household Items", "total": 386}, {"id": 71, "name": "Clean Your Rusty Garden Tools", "total": 382}, {"id": 68, "name": "Cleaning Up Branches and Twigs", "total": 291}, {"id": 13, "name": "Loading the Car", "total": 289}, {"id": 97, "name": "Set Up a Bar for a Cocktail Party", "total": 223}, {"id": 73, "name": "Organizing School Stuff", "total": 217}, {"id": 44, "name": "Chopping Wood", "total": 190}, {"id": 84, "name": "Tidying Bathroom", "total": 174}];
  var root = document.getElementById("rp");
  if (!root) return;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var video = root.querySelector(".rp-video");
  var chips = root.querySelector(".rp-chips");
  var bar = root.querySelector(".rp-bar i"), clock = root.querySelector(".rp-clock"), title = root.querySelector(".rp-title");
  var btn = root.querySelector(".rp-play"), barWrap = root.querySelector(".rp-bar");
  var cur = 0, playing = !reduce, visible = false, loadTok = 0, blobUrl = null;
  var err = document.createElement("div");
  err.className = "rp-err"; err.style.display = "none";
  root.querySelector(".rp-stage").appendChild(err);

  function src(t) { return "assets/rollouts/t" + ("000" + TASKS[t].id).slice(-3) + ".mp4?v=5"; }
  function fmt(s) { s = Math.round(s); return Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2); }

  TASKS.forEach(function (t, i) {
    var b = document.createElement("button");
    b.type = "button"; b.textContent = t.name; b.className = "rp-chip";
    b.addEventListener("click", function () { selectTask(i); });
    chips.appendChild(b);
  });

  function progress() {
    var d = video.duration || 0, k = d ? Math.min(1, video.currentTime / d) : 0;
    bar.style.width = (k * 100) + "%";
    clock.textContent = fmt(k * TASKS[cur].total);
  }

  function sync() {                        /* play only when the user wants it and the player is on screen */
    if (playing && visible) {
      var p = video.play();
      if (p && p.catch) p.catch(function (err) { if (err && err.name === "NotAllowedError") setPlaying(false); });   /* autoplay blocked: wait for a click */
    } else video.pause();
  }

  function selectTask(i) {
    cur = i;
    var cs = chips.children;
    for (var k = 0; k < cs.length; k++) cs[k].classList.toggle("on", k === i);
    title.textContent = TASKS[i].name + " \u00b7 episode length " + fmt(TASKS[i].total);
    var c = cs[i]; if (c) chips.scrollLeft = c.offsetLeft - chips.clientWidth / 2 + c.clientWidth / 2;
    loadVideo(i);
    progress();
  }

  /* Fetch the whole file with a plain GET and play it from a blob: no Range requests, so it also works behind proxies that mishandle them. */
  function loadVideo(i) {
    var tok = ++loadTok;
    err.style.display = "none";
    function direct() { if (tok !== loadTok) return; setSrc(src(i)); }
    if (!window.fetch || !window.URL || !URL.createObjectURL) { direct(); return; }
    fetch(src(i)).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.blob();
    }).then(function (b) {
      if (tok !== loadTok) return;
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      blobUrl = URL.createObjectURL(new Blob([b], { type: "video/mp4" }));
      setSrc(blobUrl);
    }).catch(direct);
  }
  function setSrc(u) { video.src = u; video.load(); progress(); sync(); }

  function setPlaying(p) {
    playing = p; btn.textContent = p ? "Pause" : "Play";
    btn.setAttribute("aria-pressed", p ? "true" : "false");
    sync();
  }

  video.addEventListener("error", function () {
    if (!video.getAttribute("src")) return;
    var c = video.error ? video.error.code : "?";
    err.innerHTML = "The video could not be played (code " + c + "). <a href=\"" + src(cur) + "\" target=\"_blank\" rel=\"noopener\">Open the file directly</a>.";
    err.style.display = "block";
  });
  video.addEventListener("timeupdate", progress);
  video.addEventListener("loadedmetadata", progress);
  video.addEventListener("ended", function () { selectTask((cur + 1) % TASKS.length); });   /* move on to the next task */
  btn.addEventListener("click", function () { setPlaying(!playing); });
  barWrap.addEventListener("click", function (e) {
    var r = barWrap.getBoundingClientRect();
    if (video.duration) { video.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * video.duration; progress(); }
  });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (en) { visible = en[0].isIntersecting; sync(); }, { threshold: 0.25 }).observe(root);
  } else { visible = true; }
  btn.textContent = playing ? "Pause" : "Play";
  selectTask(0);
})();
