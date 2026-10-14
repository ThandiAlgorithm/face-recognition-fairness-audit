// Flask serves the frontend itself (see app.py's static_folder), so the API
// lives at the same origin as the page whether that's localhost:5000 or the
// deployed Space -- no separate backend URL to configure.
const API = "/api/analyze";

const state = { name: "", photo1: null, photo2: null, result: null, ds: "fairface" };
let stream = null;

function show(name) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  const el = document.querySelector(`[data-screen="${name}"]`);
  el.classList.add("active");
  if (name === "cam1" || name === "cam2") startCam(el);
  else stopCam();
  if (name === "review") { get("rev1").src = state.photo1; get("rev2").src = state.photo2; }
  if (name === "done") { stopCam(); get("thanks").textContent = state.name ? `Thank you, ${state.name}` : "Thank you"; state.photo1 = state.photo2 = state.result = null; }
}

const get = id => document.getElementById(id);

async function startCam(screen) {
  stopCam();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    screen.querySelector("video").srcObject = stream;
  } catch (e) {
    alert("Cannot open the camera. Please allow camera access.");
  }
}
function stopCam() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } }

function capture(video) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return c.toDataURL("image/jpeg", 0.92);
}

document.querySelectorAll("[data-go]").forEach(b => b.addEventListener("click", () => show(b.dataset.go)));

get("name").addEventListener("input", e => state.name = e.target.value.trim());

function consentReady() {
  const ok = get("c-adult").checked && get("c-agree").checked && get("sign").value.trim().length > 1;
  get("consent-continue").disabled = !ok;
}
["c-adult", "c-agree", "sign"].forEach(id => get(id).addEventListener("input", consentReady));

document.querySelectorAll("[data-shoot]").forEach(btn => btn.addEventListener("click", () => {
  const screen = btn.closest(".screen");
  const shot = capture(screen.querySelector("video"));
  if (btn.dataset.shoot === "1") { state.photo1 = shot; get("thumb1").src = shot; show("cam2"); }
  else { state.photo2 = shot; show("review"); }
}));

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

document.querySelectorAll("[data-gallery]").forEach(btn => btn.addEventListener("click", () => {
  get("file" + btn.dataset.gallery).click();
}));

get("file1").addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  state.photo1 = await readFileAsDataURL(file);
  get("thumb1").src = state.photo1;
  e.target.value = "";
  show("cam2");
});

get("file2").addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  state.photo2 = await readFileAsDataURL(file);
  e.target.value = "";
  show("review");
});

const STEPS = ["detect", "align", "quality", "recog", "classify"];
function markStep(done, now) {
  document.querySelectorAll("#steps li").forEach(li => {
    li.classList.remove("now", "done");
    const i = STEPS.indexOf(li.dataset.s);
    if (i < done) li.classList.add("done");
    else if (i === now) li.classList.add("now");
  });
}

get("submit").addEventListener("click", async () => {
  show("processing");
  get("proc-err").textContent = "";
  get("proc-back").hidden = true;
  get("proc-wait").hidden = true;
  let step = 0;
  markStep(0, 0);
  const tick = setInterval(() => { step = Math.min(step + 1, STEPS.length - 1); markStep(step, step); }, 700);
  // The step list only takes ~3.5s to cycle through, but the real request
  // (DeepFace's cold-start especially) can run another 20-30s -- without
  // a live countdown, the screen looks frozen for the rest of the wait.
  // Clamped at 0:00 rather than going negative if it runs past 30s.
  const COUNTDOWN_START = 30;
  let remaining = COUNTDOWN_START;
  get("proc-timer").textContent = `0:${String(remaining).padStart(2, "0")}`;
  const timerTick = setInterval(() => {
    remaining = Math.max(0, remaining - 1);
    get("proc-timer").textContent = `0:${String(remaining).padStart(2, "0")}`;
  }, 1000);
  const waitMsg = setTimeout(() => { get("proc-wait").hidden = false; }, 5000);
  // The cold-start wait is usually 20-30s past this point -- say so, instead
  // of letting the same "still working" line sit unchanged that whole time.
  const longWaitMsg = setTimeout(() => {
    get("proc-wait").textContent = "Still working — almost there, this last stretch (model warm-up) can take up to 30 more seconds.";
  }, 30000);
  try {
    const res = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ photo1: state.photo1, photo2: state.photo2 }),
    });
    const data = await res.json();
    clearInterval(tick);
    clearInterval(timerTick);
    clearTimeout(waitMsg);
    clearTimeout(longWaitMsg);
    get("proc-wait").hidden = true;
    if (data.error) {
      markStep(0, -1);
      get("proc-err").textContent = data.error + ". Please retake your photos.";
      get("proc-back").hidden = false;
      return;
    }
    markStep(STEPS.length, -1);
    state.result = data;
    renderResults();
    setTimeout(() => show("results"), 400);
  } catch (e) {
    clearInterval(tick);
    clearInterval(timerTick);
    clearTimeout(waitMsg);
    clearTimeout(longWaitMsg);
    get("proc-wait").hidden = true;
    get("proc-err").textContent = "Could not reach the server. Is the backend running?";
    get("proc-back").hidden = false;
  }
});

document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  const v = t.dataset.view;
  get("view-you").hidden = v !== "you";
  get("view-compare").hidden = v !== "compare";
}));

document.querySelectorAll("#ds-chips .chip").forEach(c => c.addEventListener("click", () => {
  document.querySelectorAll("#ds-chips .chip").forEach(x => x.classList.remove("active"));
  c.classList.add("active");
  state.ds = c.dataset.ds;
  renderDataset();
}));

function renderRecog(key, r) {
  if (!r) { get("same-" + key).textContent = "n/a"; return; }
  get("same-" + key).textContent = r.same_person ? "Yes" : "No";
  get("score-fill-" + key).style.width = Math.round(r.score * 100) + "%";
  get("score-mark-" + key).style.left = Math.round(r.threshold * 100) + "%";
  get("score-txt-" + key).textContent = `Match score ${r.score}. Says "same person" at or above ${r.threshold}.`;
}

function renderResults() {
  get("greeting").textContent = state.name ? `Hi ${state.name}, here is what the model sees` : "Here is what the model sees";
  const rec = state.result.recognition;
  renderRecog("deepface", rec.deepface);
  renderRecog("thirdlens", rec.thirdlens);

  const attrCard = get("attr-card");
  const realChanges = (state.result.attribute_changes || []).filter(c => c.change !== "No confident change");
  if (realChanges.length) {
    attrCard.hidden = false;
    const list = get("attr-list");
    list.innerHTML = "";
    realChanges.forEach(c => {
      const li = document.createElement("li");
      li.className = "change-row";
      li.innerHTML = `<span><b>${c.attribute}</b><small>Photo 1: ${c.photo1_state} &rarr; Photo 2: ${c.photo2_state}</small></span>
        <span class="chip${c.change === "Removed" ? " orange" : ""}">${c.change}</span>`;
      list.appendChild(li);
    });
  } else {
    attrCard.hidden = true;
  }

  const body = get("cmp-body");
  body.innerHTML = "";
  state.result.diff.rows.forEach(row => {
    const tr = document.createElement("tr");
    if (row.key === state.result.diff.biggest_key) tr.className = "big";
    const f = v => v === null ? "-" : v;
    tr.innerHTML = `<td>${row.label}</td><td>${f(row.photo1)}</td><td>${f(row.photo2)}</td>`;
    body.appendChild(tr);
  });
  get("diff-msg").textContent = state.result.diff.message;
  renderDataset();
}

function renderDataset() {
  const d = state.result.photo1.datasets[state.ds];
  const meta = state.result.dataset_meta[state.ds];
  get("r-age").textContent = d.age;
  get("r-gender").textContent = d.gender;
  get("r-genconf").textContent = d.gender_conf ? `${d.gender_conf}% confident` : "";
  get("ds-title").textContent = `${meta.label} category scores`;
  get("ds-src").textContent = d.source === "custom" ? "trained model" : "general model (fallback)";

  const bars = get("race-bars");
  bars.innerHTML = "";
  if (!meta.has_race || !d.race) {
    get("race-note").textContent = `${meta.label} has no race labels, so this model reports age and gender only. This is part of the point: different datasets label people differently.`;
    return;
  }
  const entries = Object.entries(d.race).sort((a, b) => b[1] - a[1]);
  entries.forEach((e, i) => {
    const wrap = document.createElement("div");
    wrap.className = "bar" + (i === 0 ? " top" : "");
    wrap.innerHTML = `<div class="row"><span>${e[0]}</span><span>${e[1]}%</span></div>
      <div class="track"><i style="width:${Math.max(e[1], 1)}%"></i></div>`;
    bars.appendChild(wrap);
  });
  get("race-note").textContent = `These are the ${meta.label} labels. The scores show how the model reads this photo. They do not measure your ancestry or identity.`;

  const skinCard = get("skin-tone-card");
  if (meta.has_skin_tone && d.skin_tone) {
    skinCard.hidden = false;
    const skinBars = get("skin-bars");
    skinBars.innerHTML = "";
    Object.entries(d.skin_tone).sort((a, b) => b[1] - a[1]).forEach((e, i) => {
      const wrap = document.createElement("div");
      wrap.className = "bar" + (i === 0 ? " top" : "");
      wrap.innerHTML = `<div class="row"><span>${e[0]}</span><span>${e[1]}%</span></div>
        <div class="track"><i style="width:${Math.max(e[1], 1)}%"></i></div>`;
      skinBars.appendChild(wrap);
    });
  } else {
    skinCard.hidden = true;
  }
}
