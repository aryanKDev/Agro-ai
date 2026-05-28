// ═══════════════════════════════════════════════
// AgroAI — script.js  (Part 1: Utility Modules)
// ═══════════════════════════════════════════════

const API_URL = "http://127.0.0.1:5000";

// ── STORAGE UTILS ──────────────────────────────
const Storage = {
  HISTORY_KEY: "agroai_history",
  FEEDBACK_KEY: "agroai_feedback",

  // Local storage fallback methods (synchronous)
  getLocalHistory() { try { return JSON.parse(localStorage.getItem(this.HISTORY_KEY) || "[]"); } catch { return []; } },
  saveLocalHistory(arr) { localStorage.setItem(this.HISTORY_KEY, JSON.stringify(arr)); },
  addLocalScan(entry) {
    const h = this.getLocalHistory();
    // Prevent duplicate entries in local cache
    if (!h.some(item => item.id === entry.id)) {
      h.unshift(entry);
      if (h.length > 100) h.pop();
      this.saveLocalHistory(h);
    }
  },
  deleteLocalScan(id) { this.saveLocalHistory(this.getLocalHistory().filter(e => e.id !== id && e.id != id)); },
  clearLocalHistory() { localStorage.removeItem(this.HISTORY_KEY); },

  // Unified async methods that communicate with backend but fallback to local
  async getHistory() {
    try {
      const res = await fetch(API_URL + '/api/scans');
      if (!res.ok) throw new Error("Server response error");
      const data = await res.json();
      
      if (Array.isArray(data)) {
        // Sync local cache with database records
        this.saveLocalHistory(data);
        return data;
      }
    } catch (e) {
      console.warn("Failed to fetch scan history from MongoDB Atlas, using offline localStorage cache:", e);
    }
    return this.getLocalHistory();
  },

  async addScan(entry) {
    // Generate a fallback ID if backend is down and database returned None
    if (!entry.id) {
      entry.id = 'local_' + Date.now();
    }
    this.addLocalScan(entry);
  },

  async deleteScan(id) {
    // 1. Delete locally first for instant UI responsiveness
    this.deleteLocalScan(id);
    
    // 2. Sync deletion with MongoDB
    try {
      const res = await fetch(`${API_URL}/api/scans/${id}`, { method: 'DELETE' });
      if (!res.ok) console.warn("Failed to delete scan on MongoDB backend");
    } catch (e) {
      console.warn("Failed to sync delete to MongoDB Atlas:", e);
    }
  },

  async clearHistory() {
    // 1. Clear locally first
    this.clearLocalHistory();
    
    // 2. Sync with MongoDB
    try {
      const res = await fetch(API_URL + '/api/scans', { method: 'DELETE' });
      if (!res.ok) console.warn("Failed to clear scans on MongoDB backend");
    } catch (e) {
      console.warn("Failed to sync clear to MongoDB Atlas:", e);
    }
  },

  getFeedback() { try { return JSON.parse(localStorage.getItem(this.FEEDBACK_KEY) || "[]"); } catch { return []; } },
  addFeedback(entry) {
    const f = this.getFeedback();
    f.unshift({ id: Date.now(), ...entry });
    this.saveFeedback(f);
  },
  saveFeedback(arr) { localStorage.setItem(this.FEEDBACK_KEY, JSON.stringify(arr)); }
};

// ── TOAST MANAGER ──────────────────────────────
const Toast = {
  show(msg, type = "info", duration = 3500) {
    const icons = { success: "✅", error: "❌", info: "ℹ️" };
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<span class="toast-icon">${icons[type]||"ℹ️"}</span><span>${msg}</span>`;
    document.getElementById("toast-container").appendChild(el);
    setTimeout(() => { el.classList.add("toast-out"); setTimeout(() => el.remove(), 320); }, duration);
  }
};

// ── AI LOADER ──────────────────────────────────
const AILoader = {
  steps: ["🔍 Analyzing leaf patterns…","🧬 Detecting disease markers…","📊 Generating AI insights…","✅ Finalizing diagnosis…"],
  timer: null, idx: 0, barTimer: null,
  show() {
    const ov = document.getElementById("ai-loader-overlay");
    const stepEl = document.getElementById("ai-loader-step");
    const bar = document.getElementById("ai-loader-bar");
    ov.classList.remove("hidden");
    this.idx = 0; stepEl.textContent = this.steps[0]; bar.style.width = "0";
    let pct = 0;
    this.timer = setInterval(() => {
      this.idx = Math.min(this.idx + 1, this.steps.length - 1);
      stepEl.textContent = this.steps[this.idx];
    }, 1800);
    this.barTimer = setInterval(() => {
      pct = Math.min(pct + 2, 90);
      bar.style.width = pct + "%";
    }, 120);
  },
  hide() {
    clearInterval(this.timer); clearInterval(this.barTimer);
    const bar = document.getElementById("ai-loader-bar");
    bar.style.width = "100%";
    setTimeout(() => document.getElementById("ai-loader-overlay").classList.add("hidden"), 400);
  }
};

// ── PAGE ROUTER ────────────────────────────────
const Router = {
  pages: ["home", "history", "analytics", "feedback"],
  current: "home",
  navigate(page) {
    this.pages.forEach(p => {
      const el = document.getElementById("page-" + p);
      if (el) el.classList.toggle("hidden", p !== page);
    });
    document.querySelectorAll(".nav-page-link").forEach(a => {
      a.classList.toggle("active-nav", a.dataset.page === page);
    });
    document.getElementById("nav-links").classList.remove("open");
    this.current = page;
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (page === "history") HistoryManager.render();
    if (page === "analytics") Analytics.render();
    if (page === "feedback") Feedback.render();
  }
};

// ── SEVERITY UTILS ─────────────────────────────
const Severity = {
  get(confidence) {
    if (confidence >= 90) return { label: "⚠️ HIGH RISK", cls: "sev-high" };
    if (confidence >= 70) return { label: "🟡 MEDIUM RISK", cls: "sev-medium" };
    return { label: "🟢 LOW RISK", cls: "sev-low" };
  }
};

// ── PDF GENERATOR (Backend-Powered — Premium ReportLab PDF) ──────────────
const PDF = {
  async generate() {
    const btn = document.getElementById("download-pdf-btn");
    const originalText = btn.textContent;
    btn.innerHTML = `<span class="btn-label">⏳ Generating Premium Report…</span>`;
    btn.disabled = true;

    try {
      // ── Gather all scan data from the current result ──────────────────
      const disease      = document.getElementById("prediction-output")?.textContent?.trim() || "Unknown";
      const confidenceEl = document.getElementById("confidence-value");
      const confidence   = parseFloat(confidenceEl?.textContent || "0");
      const severityEl   = document.getElementById("severity-badge");
      const severityRaw  = severityEl?.textContent || "";

      // Map severity badge text to API value
      let severity = "LOW";
      if (severityRaw.includes("HIGH"))   severity = "HIGH";
      else if (severityRaw.includes("MEDIUM")) severity = "MEDIUM";

      const symptoms   = document.getElementById("symptoms-output")?.innerText?.trim()   || "";
      const treatment  = document.getElementById("treatment-output")?.innerText?.trim()  || "";
      const prevention = document.getElementById("prevention-output")?.innerText?.trim() || "";

      // Detect healthy state
      const isHealthy = !!(lastResult?.isHealthy);

      // Get image data URL from preview (if exists)
      const imgEl = document.querySelector("#image-preview img");
      const imageDataUrl = (imgEl && imgEl.src && imgEl.src.startsWith("data:")) ? imgEl.src : null;

      // Scan & DB IDs from last result
      const scanId  = lastResult?.id || ("SCAN-" + Date.now());
      const dbId    = (lastResult?.id && !String(lastResult.id).startsWith("local_")) ? lastResult.id : null;
      const fname   = lastResult?.filename || currentFile?.name || "upload.jpg";

      // ── Send to backend /generate-report ─────────────────────────────
      const payload = {
        disease_name:    disease,
        confidence:      confidence,
        symptoms:        symptoms,
        treatment:       treatment,
        prevention:      prevention,
        severity:        severity,
        is_healthy:      isHealthy,
        scan_id:         scanId,
        db_id:           dbId,
        filename:        fname,
        image_data_url:  imageDataUrl,
        plant_type:      disease.split(" ")[0] || "Plant",
      };

      const res = await fetch(API_URL + "/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || `Server error ${res.status}`);
      }

      // ── Trigger file download from binary blob ────────────────────────
      const blob     = await res.blob();
      const url      = URL.createObjectURL(blob);
      const safeName = disease.replace(/[^a-z0-9]/gi, "_").slice(0, 40);
      const link     = document.createElement("a");
      link.href      = url;
      link.download  = `AgroAI_Report_${safeName}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(url), 5000);

      Toast.show("✅ Premium AI Report downloaded successfully!", "success");

    } catch (e) {
      console.error("PDF generation error:", e);
      Toast.show("PDF generation failed: " + e.message, "error");
    } finally {
      btn.innerHTML = `<span class="btn-label">${originalText}</span>`;
      btn.disabled  = false;
    }
  }
};

// ── HISTORY MANAGER ────────────────────────────
const HistoryManager = {
  async render() {
    const grid = document.getElementById("history-grid");
    const empty = document.getElementById("history-empty");
    const meta = document.getElementById("history-meta");
    const items = await Storage.getHistory();
    if (!items.length) {
      grid.innerHTML = ""; empty.classList.remove("hidden"); meta.textContent = "";
      return;
    }
    empty.classList.add("hidden");
    meta.textContent = `${items.length} scan${items.length !== 1 ? "s" : ""} saved`;
    grid.innerHTML = items.map(item => {
      const imgPart = item.imageDataUrl
        ? `<img class="history-card-img" src="${item.imageDataUrl}" alt="${item.disease}" loading="lazy">`
        : `<div class="history-card-img-placeholder">🌿</div>`;
      const sevCls = item.severity === "HIGH" ? "sev-high" : item.severity === "MEDIUM" ? "sev-medium" : "sev-low";
      
      // Support both MongoDB ISO string timestamp and local millisecond ID
      const timestampSource = item.timestamp || item.id;
      const date = isNaN(Number(timestampSource)) ? new Date(timestampSource).toLocaleString() : new Date(Number(timestampSource)).toLocaleString();
      
      return `<div class="history-card" id="hcard-${item.id}">
        ${imgPart}
        <div class="history-card-body">
          <div class="history-card-disease">${item.disease}</div>
          <div class="history-card-row">
            <span class="severity-badge ${sevCls}">${item.severity || "N/A"}</span>
            <span class="confidence-value">${item.confidence ? item.confidence + "%" : "—"}</span>
          </div>
          <div class="history-card-time">${date}</div>
        </div>
        <div class="history-card-footer">
          <span style="color:var(--text2);font-size:.75rem">📋 Scan #${item.id.toString().slice(-6)}</span>
          <button class="history-delete-btn" onclick="HistoryManager.delete('${item.id}')">🗑️ Delete</button>
        </div>
      </div>`;
    }).join("");
  },
  async delete(id) {
    await Storage.deleteScan(id);
    const card = document.getElementById("hcard-" + id);
    if (card) { card.style.opacity = "0"; card.style.transform = "scale(0.9)"; setTimeout(() => this.render(), 300); }
    Toast.show("Scan deleted.", "info");
  },
  async clearAll() {
    if (!confirm("Delete all scan history? This cannot be undone.")) return;
    await Storage.clearHistory(); await this.render();
    Toast.show("All history cleared.", "info");
  }
};

// ── ANALYTICS ──────────────────────────────────
const Analytics = {
  charts: {},
  async render() {
    const items = await Storage.getHistory();
    const empty = document.getElementById("analytics-empty");
    if (!items.length) { empty.classList.remove("hidden"); return; }
    empty.classList.add("hidden");
    const diseased = items.filter(i => i.isHealthy === false);
    const healthy = items.filter(i => i.isHealthy === true);
    document.getElementById("kpi-total").textContent = items.length;
    document.getElementById("kpi-diseased").textContent = diseased.length;
    document.getElementById("kpi-healthy").textContent = healthy.length;
    const freq = {};
    diseased.forEach(i => { freq[i.disease] = (freq[i.disease] || 0) + 1; });
    const topDisease = Object.keys(freq).sort((a, b) => freq[b] - freq[a])[0] || "—";
    document.getElementById("kpi-top").textContent = topDisease;
    this.renderDonut(healthy.length, diseased.length);
    this.renderBar(freq);
    this.renderLine(items);
  },
  destroy(id) { if (this.charts[id]) { this.charts[id].destroy(); delete this.charts[id]; } },
  chartDefaults() {
    return { color: "#e2e8f0", borderColor: "rgba(255,255,255,0.1)",
      plugins: { legend: { labels: { color: "#94a3b8", font: { family: "Inter" } } } },
      scales: { x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } },
                y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } } }
    };
  },
  renderDonut(h, d) {
    this.destroy("donut");
    const ctx = document.getElementById("chart-donut");
    if (!ctx) return;
    this.charts.donut = new Chart(ctx, {
      type: "doughnut",
      data: { labels: ["Healthy", "Diseased"], datasets: [{ data: [h || 0, d || 0], backgroundColor: ["rgba(16,185,129,0.7)", "rgba(239,68,68,0.7)"], borderColor: ["rgba(16,185,129,1)", "rgba(239,68,68,1)"], borderWidth: 2 }] },
      options: { plugins: { legend: { labels: { color: "#94a3b8" } } }, cutout: "65%" }
    });
  },
  renderBar(freq) {
    this.destroy("bar");
    const ctx = document.getElementById("chart-bar");
    if (!ctx) return;
    const labels = Object.keys(freq).slice(0, 8).map(k => k.length > 16 ? k.slice(0, 16) + "…" : k);
    const values = Object.values(freq).slice(0, 8);
    const def = this.chartDefaults();
    this.charts.bar = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets: [{ label: "Occurrences", data: values, backgroundColor: "rgba(124,58,237,0.6)", borderColor: "rgba(124,58,237,1)", borderWidth: 1, borderRadius: 6 }] },
      options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
    });
  },
  renderLine(items) {
    this.destroy("line");
    const ctx = document.getElementById("chart-line");
    if (!ctx) return;
    const days = {}; const now = Date.now();
    for (let i = 6; i >= 0; i--) {
      const d = new Date(now - i * 86400000);
      days[d.toLocaleDateString("en", { month: "short", day: "numeric" })] = 0;
    }
    items.forEach(item => {
      const ts = item.timestamp || item.id;
      const parsedTs = isNaN(Number(ts)) ? ts : Number(ts);
      const d = new Date(parsedTs).toLocaleDateString("en", { month: "short", day: "numeric" });
      if (d in days) days[d]++;
    });
    const def = this.chartDefaults();
    this.charts.line = new Chart(ctx, {
      type: "line",
      data: { labels: Object.keys(days), datasets: [{ label: "Scans", data: Object.values(days), borderColor: "rgba(16,185,129,1)", backgroundColor: "rgba(16,185,129,0.1)", borderWidth: 2, tension: 0.4, fill: true, pointBackgroundColor: "rgba(16,185,129,1)" }] },
      options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
    });
  }
};

// ── FEEDBACK ───────────────────────────────────
const Feedback = {
  selected: 0,
  init() {
    document.querySelectorAll(".star").forEach(s => {
      s.addEventListener("mouseenter", () => this.highlight(+s.dataset.val));
      s.addEventListener("mouseleave", () => this.highlight(this.selected));
      s.addEventListener("click", () => { this.selected = +s.dataset.val; this.highlight(this.selected); document.getElementById("star-label").textContent = ["", "Poor","Fair","Good","Great","Excellent!"][this.selected]; });
    });
    document.getElementById("feedback-submit-btn").addEventListener("click", () => this.submit());
    document.getElementById("go-scan-btn")?.addEventListener("click", () => Router.navigate("home"));
    document.getElementById("go-scan-analytics-btn")?.addEventListener("click", () => Router.navigate("home"));
  },
  highlight(val) { document.querySelectorAll(".star").forEach(s => s.classList.toggle("active", +s.dataset.val <= val)); },
  submit() {
    if (!this.selected) { Toast.show("Please select a star rating.", "error"); return; }
    const text = document.getElementById("feedback-text").value.trim();
    Storage.addFeedback({ rating: this.selected, text, timestamp: new Date().toLocaleString() });
    this.selected = 0; this.highlight(0);
    document.getElementById("feedback-text").value = "";
    document.getElementById("star-label").textContent = "Click to rate";
    Toast.show("Thank you for your feedback! ⭐", "success");
    this.render();
  },
  render() {
    const items = Storage.getFeedback();
    const list = document.getElementById("feedback-list");
    const avg = document.getElementById("avg-rating");
    const total = document.getElementById("total-feedback");
    total.textContent = items.length;
    avg.textContent = items.length ? (items.reduce((s, i) => s + i.rating, 0) / items.length).toFixed(1) + " ⭐" : "—";
    if (!list) return;
    list.innerHTML = items.slice(0, 10).map(item => `
      <div class="feedback-item">
        <div class="feedback-item-stars">${"★".repeat(item.rating)}${"☆".repeat(5 - item.rating)}</div>
        ${item.text ? `<div class="feedback-item-text">"${item.text}"</div>` : ""}
        <div class="feedback-item-time">${item.timestamp}</div>
      </div>`).join("");
  }
};

// ── VOICE INPUT ────────────────────────────────
const Voice = {
  recog: null,
  init() {
    const btn = document.getElementById("voice-btn");
    if (!btn) return;
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) { btn.title = "Voice not supported in this browser"; btn.style.opacity = "0.4"; return; }
    this.recog = new SpeechRec();
    this.recog.continuous = false; this.recog.interimResults = false; this.recog.lang = "en-US";
    this.recog.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      document.getElementById("chat-message").value = transcript;
      btn.classList.remove("listening");
      Toast.show(`Voice: "${transcript}"`, "info");
    };
    this.recog.onerror = () => { btn.classList.remove("listening"); Toast.show("Voice recognition error.", "error"); };
    this.recog.onend = () => btn.classList.remove("listening");
    btn.addEventListener("click", () => {
      if (btn.classList.contains("listening")) { this.recog.stop(); btn.classList.remove("listening"); }
      else { this.recog.start(); btn.classList.add("listening"); Toast.show("Listening… speak now 🎤", "info"); }
    });
  }
};

// ── CAMERA CAPTURE ─────────────────────────────
const Camera = {
  init() {
    const input = document.getElementById("camera-capture");
    if (!input) return;
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (file) { handleFile(file); input.value = ""; }
    });
  }
};

// =============================================
// Part 2: Core App Logic
// =============================================
const imageUploader  = document.getElementById('image-uploader');
const urlUploader    = document.getElementById('url-uploader');
const predictButton  = document.getElementById('predict-button');
const imagePreview   = document.getElementById('image-preview');
const preventionCard = document.getElementById('prevention-card');
const resultsDiv     = document.getElementById('results');
const predictionOutput = document.getElementById('prediction-output');
const symptomsOutput   = document.getElementById('symptoms-output');
const treatmentOutput  = document.getElementById('treatment-output');
const preventionOutput = document.getElementById('prevention-output');
const symptomsSection  = document.getElementById('symptoms-section');
const treatmentSection = document.getElementById('treatment-section');
const preventionTitle  = document.getElementById('prevention-title');
const chatContainer    = document.getElementById('chat-container');
const chatBox          = document.getElementById('chat-box');
const chatInput        = document.getElementById('chat-message');
const chatSendButton   = document.getElementById('chat-send-button');
const uploadArea       = document.getElementById('upload-area');
const uploadLabel      = document.querySelector('.upload-label');
let chatContext = { session_id: 'user_' + Date.now(), disease: '', symptoms: '', treatment: '', prevention: '' };
let currentFile = null, lastResult = null;
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => navbar.classList.toggle('scrolled', window.scrollY > 20));
const animObs = new IntersectionObserver((entries) => { entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }); }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
document.querySelectorAll('.anim-fade-up').forEach(el => animObs.observe(el));
function animateCounter(el) { const target = parseInt(el.dataset.target), suffix = el.dataset.suffix || ''; let cur = 0; const step = Math.ceil(target / 60); const t = setInterval(() => { cur = Math.min(cur + step, target); el.textContent = cur + suffix; if (cur >= target) clearInterval(t); }, 20); }
const statObs = new IntersectionObserver((entries) => { entries.forEach(e => { if (e.isIntersecting && !e.target.dataset.counted) { e.target.dataset.counted = '1'; animateCounter(e.target); } }); }, { threshold: 0.5 });
document.querySelectorAll('.stat-number').forEach(el => statObs.observe(el));
(function initParticles() { const canvas = document.getElementById('particles-canvas'); if (!canvas) return; const ctx = canvas.getContext('2d'); let W, H, particles = []; const resize = () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }; resize(); window.addEventListener('resize', resize); const colors = ['rgba(124,58,237,0.5)','rgba(16,185,129,0.4)','rgba(59,130,246,0.4)','rgba(168,85,247,0.4)']; for (let i = 0; i < 55; i++) particles.push({ x:Math.random()*window.innerWidth, y:Math.random()*window.innerHeight, r:Math.random()*1.8+0.4, dx:(Math.random()-0.5)*0.4, dy:(Math.random()-0.5)*0.4, color:colors[Math.floor(Math.random()*colors.length)] }); (function draw() { ctx.clearRect(0,0,W,H); particles.forEach(p => { ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fillStyle=p.color; ctx.fill(); p.x+=p.dx; p.y+=p.dy; if(p.x<0||p.x>W)p.dx*=-1; if(p.y<0||p.y>H)p.dy*=-1; }); requestAnimationFrame(draw); })(); })();
document.addEventListener('mousemove', (e) => { document.documentElement.style.setProperty('--mx', e.clientX+'px'); document.documentElement.style.setProperty('--my', e.clientY+'px'); });
document.getElementById('hamburger').addEventListener('click', () => document.getElementById('nav-links').classList.toggle('open'));
document.querySelectorAll('[data-page]').forEach(el => { el.addEventListener('click', (e) => { e.preventDefault(); Router.navigate(el.dataset.page); }); });
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => { e.preventDefault(); uploadArea.classList.remove('dragover'); const file = e.dataTransfer.files[0]; if (file && file.type.startsWith('image/')) handleFile(file); });
imageUploader.addEventListener('change', () => { if (imageUploader.files[0]) handleFile(imageUploader.files[0]); });
function handleFile(file) { currentFile = file; urlUploader.value = ''; const reader = new FileReader(); reader.onload = (e) => { imagePreview.innerHTML = '<img src=' + e.target.result + ' alt=Preview>'; imagePreview.style.display = 'block'; }; reader.readAsDataURL(file); if (uploadLabel) uploadLabel.textContent = file.name; }
urlUploader.addEventListener('input', () => { const url = urlUploader.value.trim(); if (url) { currentFile = null; imageUploader.value = ''; if (uploadLabel) uploadLabel.textContent = 'Click to upload or drag & drop'; imagePreview.innerHTML = '<img src=' + url + ' alt=Preview onerror=this.parentElement.style.display=\'none\'>'; imagePreview.style.display = 'block'; } else { imagePreview.innerHTML = ''; imagePreview.style.display = 'none'; } });
predictButton.addEventListener('click', async () => { const imageUrl = urlUploader.value.trim(); let requestBody, requestHeaders = {}; if (currentFile) { const fd = new FormData(); fd.append('file', currentFile); requestBody = fd; } else if (imageUrl) { requestBody = JSON.stringify({ url: imageUrl }); requestHeaders['Content-Type'] = 'application/json'; } else { Toast.show('Please choose an image file or paste a URL.', 'error'); return; } setLoading(true); AILoader.show(); preventionCard.classList.add('hidden'); resultsDiv.classList.add('hidden'); chatContainer.classList.add('hidden'); try { const res = await fetch(API_URL + '/predict', { method: 'POST', headers: requestHeaders, body: requestBody }); if (!res.ok) { const err = await res.json(); throw new Error(err.error || 'Prediction failed'); } const data = await res.json(); AILoader.hide(); displayPrediction(data); chatContext = { ...chatContext, disease: data.disease_name, symptoms: data.symptoms, treatment: data.treatment, prevention: data.prevention }; chatBox.innerHTML = ''; addMessageToChat('Welcome! I detected **' + data.disease_name + '**.\n\nAsk me about:\n- **Symptoms**\n- **Treatment**\n- **Prevention**\n\n_I am here to help!_', 'bot'); } catch (err) { AILoader.hide(); Toast.show('Error: ' + err.message, 'error'); preventionCard.classList.remove('hidden'); } finally { setLoading(false); } });
function setLoading(on) { const lbl = predictButton.querySelector('.btn-label'), sp = document.getElementById('btn-spinner'); predictButton.disabled = on; if (lbl) lbl.textContent = on ? 'Analyzing...' : 'Analyze Image'; if (sp) sp.classList.toggle('hidden', !on); predictButton.classList.toggle('loading', on); }
function displayPrediction(data) { predictionOutput.textContent = data.disease_name; symptomsOutput.innerHTML = (data.symptoms || '').replace(/\n/g, '<br>'); treatmentOutput.innerHTML = (data.treatment || '').replace(/\n/g, '<br>'); preventionOutput.innerHTML = (data.prevention || '').replace(/\n/g, '<br>'); const isHealthy = data.prediction.includes('healthy'); const confidence = typeof data.confidence === 'number' ? data.confidence : (isHealthy ? 97 : 92); const sev = Severity.get(confidence); const bar = document.getElementById('confidence-bar'), val = document.getElementById('confidence-value'), badge = document.getElementById('severity-badge'), warn = document.getElementById('severity-warning'); if (bar) setTimeout(() => { bar.style.width = confidence + '%'; }, 300); if (val) val.textContent = confidence + '%'; if (isHealthy) { predictionOutput.classList.add('healthy'); symptomsSection.classList.add('hidden'); treatmentSection.classList.add('hidden'); preventionTitle.textContent = 'How to Keep it Healthy'; if (badge) { badge.textContent = 'Healthy'; badge.className = 'severity-badge sev-low'; } if (warn) warn.classList.add('hidden'); } else { predictionOutput.classList.remove('healthy'); symptomsSection.classList.remove('hidden'); treatmentSection.classList.remove('hidden'); preventionTitle.textContent = 'Prevention'; if (badge) { badge.textContent = sev.label; badge.className = 'severity-badge ' + sev.cls; } if (warn) warn.classList.toggle('hidden', sev.cls !== 'sev-high'); } resultsDiv.classList.remove('hidden'); chatContainer.classList.remove('hidden'); resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' }); const imgEl = document.querySelector('#image-preview img'); let imageDataUrl = null; if (imgEl && imgEl.src && imgEl.src.startsWith('data:')) imageDataUrl = imgEl.src; lastResult = { id: data.id, imageDataUrl, disease: data.disease_name, confidence, severity: isHealthy ? 'LOW' : (sev.cls === 'sev-high' ? 'HIGH' : sev.cls === 'sev-medium' ? 'MEDIUM' : 'LOW'), isHealthy }; Storage.addScan(lastResult); Toast.show('Scan saved: ' + data.disease_name + ' (' + confidence + '%)', 'success'); }
document.getElementById('download-pdf-btn') && document.getElementById('download-pdf-btn').addEventListener('click', () => PDF.generate());
document.getElementById('save-history-manual-btn') && document.getElementById('save-history-manual-btn').addEventListener('click', () => { if (lastResult) { Storage.addScan(lastResult); Toast.show('Saved!', 'success'); } else Toast.show('No result yet.', 'info'); });
document.getElementById('clear-all-btn') && document.getElementById('clear-all-btn').addEventListener('click', () => HistoryManager.clearAll());
chatSendButton.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keyup', (e) => { if (e.key === 'Enter') sendChatMessage(); });
async function sendChatMessage() { const message = chatInput.value.trim(); if (!message) return; addMessageToChat(message, 'user'); chatInput.value = ''; chatSendButton.disabled = true; const typing = createTypingIndicator(); chatBox.appendChild(typing); chatBox.scrollTop = chatBox.scrollHeight; try { const res = await fetch(API_URL + '/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, session_id: chatContext.session_id, disease: chatContext.disease, symptoms: chatContext.symptoms, treatment: chatContext.treatment, prevention: chatContext.prevention }) }); if (chatBox.contains(typing)) chatBox.removeChild(typing); if (!res.ok) throw new Error('Chat failed'); const data = await res.json(); const isLocal = data.mode === 'local'; if (isLocal) showLocalModeBadge(); addMessageToChat(data.response, 'bot', isLocal); } catch { if (chatBox.contains(typing)) chatBox.removeChild(typing); showLocalModeBadge(); addMessageToChat('Using offline mode. Ask me about symptoms, treatment or prevention!', 'bot', true); } finally { chatSendButton.disabled = false; chatInput.focus(); } }
function showLocalModeBadge() { const b = document.getElementById('local-mode-badge'); if (b) b.classList.remove('hidden'); }
function createTypingIndicator() { const w = document.createElement('div'); w.classList.add('chat-msg','bot','typing-indicator'); w.innerHTML = '<span class=typing-label>Expert is thinking</span><span class=dots><span></span><span></span><span></span></span>'; return w; }
function addMessageToChat(text, sender, isLocalMode = false) { const div = document.createElement('div'); div.classList.add('chat-msg', sender); if (sender === 'bot' && isLocalMode) div.classList.add('local-mode'); let html = text.replace(/\*\*(.*?)\*\*/g,'<strong></strong>').replace(/_(.*?)_/g,'<em></em>').replace(/^[-]\s(.+)/gm,'<li></li>').replace(/(<li>.*<\/li>\n?)+/g,m=>'<ul>'+m+'</ul>').replace(/\n/g,'<br>'); if (sender === 'bot' && isLocalMode) html = '<div class=local-badge>Local Expert Mode</div>' + html; div.innerHTML = html; chatBox.appendChild(div); chatBox.scrollTop = chatBox.scrollHeight; }
Voice.init(); Camera.init(); Feedback.init();
window.AgroAI = { PDF, Router, Toast, HistoryManager, Analytics, Feedback, Storage };
