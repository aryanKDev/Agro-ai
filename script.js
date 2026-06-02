// ═══════════════════════════════════════════════
// AgroAI — script.js  (Phase 2A/2B/2C Upgrade)
// ═══════════════════════════════════════════════

const API_URL = "http://127.0.0.1:5000";

// ── I18n MODULE (Phase 2A) ─────────────────────
const I18n = {
  _lang: 'en',
  _strings: {},
  _cache: {},

  async init() {
    const saved = localStorage.getItem('agroai_lang') || 'en';
    await this.setLang(saved, false);
    this._bindToggle();
  },

  async setLang(lang, persist = true) {
    if (!this._cache[lang]) {
      try {
        const res = await fetch(`/translations/${lang}.json`);
        this._cache[lang] = await res.json();
      } catch (e) {
        console.warn('[I18n] Failed to load', lang, e);
        return;
      }
    }
    this._lang = lang;
    this._strings = this._cache[lang];
    if (persist) localStorage.setItem('agroai_lang', lang);
    this._apply();
    this._updateToggle();
    // Update voice recognition language
    if (window.Voice && Voice.recog) Voice.recog.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
  },

  t(key, fallback) {
    return this._strings[key] || fallback || key;
  },

  getLang() { return this._lang; },

  _apply() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      if (this._strings[key]) el.textContent = this._strings[key];
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      if (this._strings[key]) el.placeholder = this._strings[key];
    });
  },

  _bindToggle() {
    document.querySelectorAll('[data-lang]').forEach(btn => {
      btn.addEventListener('click', () => this.setLang(btn.dataset.lang));
    });
  },

  _updateToggle() {
    document.querySelectorAll('[data-lang]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === this._lang);
    });
  }
};

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
      const headers = window.Auth ? Auth.getAuthHeaders() : {};
      const token = window.Auth ? Auth.getToken() : null;
      console.log('[Storage.getHistory] isLoggedIn:', window.Auth ? Auth.isLoggedIn() : false, '| hasToken:', !!token);
      const res = await fetch(API_URL + '/api/scans', { headers });
      if (!res.ok) throw new Error('Server error ' + res.status);
      const data = await res.json();
      console.log('[Storage.getHistory] Server returned', data.length, 'scans');
      if (Array.isArray(data) && data.length > 0) {
        // Only overwrite local cache when server actually has records
        this.saveLocalHistory(data);
        return data;
      }
      if (Array.isArray(data) && data.length === 0) {
        // Server returned empty — user has no cloud scans yet
        // Return empty (not local cache) so UI shows correct state
        console.log('[Storage.getHistory] Server has 0 records for this user/session');
        return [];
      }
    } catch (e) {
      console.warn('[Storage.getHistory] Backend unreachable, using localStorage cache:', e.message);
    }
    // Fallback: return localStorage cache
    const local = this.getLocalHistory();
    console.log('[Storage.getHistory] Returning', local.length, 'cached records from localStorage');
    return local;
  },

  async addScan(entry) {
    // Generate a fallback ID if backend is down and database returned None
    if (!entry.id) {
      entry.id = 'local_' + Date.now();
    }
    // Always tag with timestamp for sorting
    if (!entry.timestamp) entry.timestamp = new Date().toISOString();
    this.addLocalScan(entry);
  },

  async deleteScan(id) {
    this.deleteLocalScan(id);
    try {
      const headers = window.Auth ? Auth.getAuthHeaders() : {};
      const res = await fetch(`${API_URL}/api/scans/${id}`, { method: 'DELETE', headers });
      if (!res.ok) console.warn('[Storage.deleteScan] Backend delete failed:', res.status);
    } catch (e) {
      console.warn('[Storage.deleteScan] Failed to sync delete:', e.message);
    }
  },

  async clearHistory() {
    this.clearLocalHistory();
    try {
      const headers = window.Auth ? Auth.getAuthHeaders() : {};
      const res = await fetch(API_URL + '/api/scans', { method: 'DELETE', headers });
      if (!res.ok) console.warn('[Storage.clearHistory] Backend clear failed:', res.status);
    } catch (e) {
      console.warn('[Storage.clearHistory] Failed to sync clear:', e.message);
    }
  },

  getFeedback() { try { return JSON.parse(localStorage.getItem(this.FEEDBACK_KEY) || '[]'); } catch { return []; } },
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
  pages: ["home", "history", "analytics", "feedback", "profile"],
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
    if (page === "profile") ProfilePage.render();
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
    const grid  = document.getElementById('history-grid');
    const empty = document.getElementById('history-empty');
    const meta  = document.getElementById('history-meta');
    const items = await Storage.getHistory();

    console.log('[HistoryManager.render] items:', items.length,
                '| loggedIn:', window.Auth ? Auth.isLoggedIn() : false);

    if (!items.length) {
      grid.innerHTML = '';
      // Context-aware empty state
      if (window.Auth && Auth.isLoggedIn()) {
        empty.querySelector('p').textContent =
          'Scan a plant image while logged in to save scans to your account.';
      } else {
        empty.querySelector('p').textContent =
          'Your scan history will appear here after you analyze your first plant image.';
      }
      empty.classList.remove('hidden');
      meta.textContent = '';
      return;
    }

    empty.classList.add('hidden');
    meta.textContent = `${items.length} scan${items.length !== 1 ? 's' : ''} saved`;

    const riskColor = { HIGH: 'sev-high', MEDIUM: 'sev-medium', LOW: 'sev-low' };

    grid.innerHTML = items.map(item => {
      const imgPart = item.imageDataUrl
        ? `<img class="history-card-img" src="${item.imageDataUrl}" alt="${item.disease}" loading="lazy">`
        : `<div class="history-card-img-placeholder">🌿</div>`;
      const sevCls = item.severity === 'HIGH' ? 'sev-high' : item.severity === 'MEDIUM' ? 'sev-medium' : 'sev-low';
      const rl     = item.riskLevel || '';
      const riskBadge = rl
        ? `<span class="severity-badge ${riskColor[rl] || 'sev-low'}" title="Spread Risk">${rl} RISK</span>`
        : '';

      // Support both MongoDB ISO string timestamp and local millisecond ID
      const timestampSource = item.timestamp || item.id;
      const date = isNaN(Number(timestampSource))
        ? new Date(timestampSource).toLocaleString()
        : new Date(Number(timestampSource)).toLocaleString();

      return `<div class="history-card" id="hcard-${item.id}">
        ${imgPart}
        <div class="history-card-body">
          <div class="history-card-disease">${item.disease}</div>
          <div class="history-card-row">
            <span class="severity-badge ${sevCls}">${item.severity || 'N/A'}</span>
            <span class="confidence-value">${item.confidence ? item.confidence + '%' : '—'}</span>
          </div>
          ${riskBadge ? `<div class="history-card-row" style="margin-top:4px">${riskBadge}</div>` : ''}
          <div class="history-card-time">${date}</div>
        </div>
        <div class="history-card-footer">
          <span style="color:var(--text2);font-size:.75rem">📋 Scan #${item.id.toString().slice(-6)}</span>
          <button class="history-delete-btn" onclick="HistoryManager.delete('${item.id}')">🗑️ Delete</button>
        </div>
      </div>`;
    }).join('');
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

// ── ANALYTICS (Phase 2C Enhanced) ─────────────────────────
const Analytics = {
  charts: {},
  async render() {
    const items = await Storage.getHistory();
    const empty = document.getElementById('analytics-empty');

    // Try to load live dashboard stats for authenticated users
    let dashStats = null;
    if (window.Auth && Auth.isLoggedIn()) {
      try {
        const res = await fetch(API_URL + '/api/dashboard', { headers: { ...Auth.getAuthHeaders() } });
        if (res.ok) dashStats = await res.json();
      } catch (_) {}
    }

    const total    = dashStats ? dashStats.totalScans    : items.length;
    const diseased = dashStats ? dashStats.diseasedPlants: items.filter(i => !i.isHealthy).length;
    const healthy  = dashStats ? dashStats.healthyPlants : items.filter(i => i.isHealthy).length;

    if (!total && !items.length) { empty?.classList.remove('hidden'); return; }
    empty?.classList.add('hidden');

    const setEl = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setEl('kpi-total',    total);
    setEl('kpi-diseased', diseased);
    setEl('kpi-healthy',  healthy);

    // Last scan date
    if (dashStats?.lastScan?.date || items.length) {
      const lastDate = dashStats?.lastScan?.date || items[0]?.timestamp;
      setEl('kpi-last-scan', lastDate ? new Date(lastDate).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }) : '—');
    }

    // Phase 2C KPI: Highest Risk Scan
    const hrScan = dashStats?.highestRiskScan;
    if (hrScan) {
      setEl('kpi-highest-risk', hrScan.disease.split('___').pop().replace(/_/g,' ').slice(0,18) + ' (' + hrScan.riskScore + '%)');
    } else {
      const hrLocal = items.reduce((best, i) => (i.riskScore || 0) > (best?.riskScore || 0) ? i : best, null);
      setEl('kpi-highest-risk', hrLocal ? hrLocal.disease.split('___').pop().replace(/_/g,' ').slice(0,18) : '—');
    }

    // Phase 2C KPI: Avg Confidence
    const avgConf = dashStats?.avgConfidence ?? (items.length ? Math.round(items.reduce((s,i)=>s+(i.confidence||0),0)/items.length) : null);
    setEl('kpi-avg-conf', avgConf !== null ? avgConf + '%' : '—');

    // Disease freq for bar chart
    const freq = {};
    items.filter(i => !i.isHealthy).forEach(i => { freq[i.disease] = (freq[i.disease] || 0) + 1; });

    // Charts
    this.renderDonut(healthy, diseased);
    this.renderBar(freq, items);
    this.renderLine(dashStats?.scanActivityTrend, items);
    this.renderRisk(dashStats?.riskBreakdown, items);
    this.renderMonthly(dashStats?.monthlyProgress);

    // Activity timeline
    const activity = dashStats?.recentActivity || items.slice(0, 10).map(i => ({
      id: i.id, disease: i.disease, isHealthy: i.isHealthy,
      severity: i.severity, riskLevel: i.riskLevel || 'LOW', date: i.timestamp,
    }));
    this.renderActivity(activity);
  },

  destroy(id) { if (this.charts[id]) { this.charts[id].destroy(); delete this.charts[id]; } },

  chartDefaults() {
    return {
      color: '#e2e8f0', borderColor: 'rgba(255,255,255,0.1)',
      plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } } },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } }
      }
    };
  },

  renderDonut(h, d) {
    this.destroy('donut');
    const ctx = document.getElementById('chart-donut');
    if (!ctx) return;
    this.charts.donut = new Chart(ctx, {
      type: 'doughnut',
      data: { labels: ['Healthy', 'Diseased'], datasets: [{ data: [h || 0, d || 0], backgroundColor: ['rgba(16,185,129,0.7)', 'rgba(239,68,68,0.7)'], borderColor: ['rgba(16,185,129,1)', 'rgba(239,68,68,1)'], borderWidth: 2 }] },
      options: { plugins: { legend: { labels: { color: '#94a3b8' } } }, cutout: '65%', responsive: true, maintainAspectRatio: false }
    });
  },

  renderBar(freq, items) {
    this.destroy('bar');
    const ctx = document.getElementById('chart-bar');
    if (!ctx) return;
    // Build frequency if not provided
    if (!Object.keys(freq).length && items.length) {
      items.filter(i => !i.isHealthy).forEach(i => { freq[i.disease] = (freq[i.disease] || 0) + 1; });
    }
    const labels = Object.keys(freq).slice(0, 8).map(k => k.split('___').pop().replace(/_/g, ' ').slice(0, 16));
    const values = Object.values(freq).slice(0, 8);
    const def = this.chartDefaults();
    this.charts.bar = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [{ label: 'Occurrences', data: values, backgroundColor: 'rgba(124,58,237,0.6)', borderColor: 'rgba(124,58,237,1)', borderWidth: 1, borderRadius: 6 }] },
      options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
    });
  },

  renderLine(trendData, items) {
    this.destroy('line');
    const ctx = document.getElementById('chart-line');
    if (!ctx) return;
    let labels, values;
    if (trendData && trendData.length) {
      labels = trendData.map(d => d.date);
      values = trendData.map(d => d.count);
    } else {
      // Fallback: compute 7-day from local items
      const days = {}; const now = Date.now();
      for (let i = 6; i >= 0; i--) {
        const d = new Date(now - i * 86400000);
        days[d.toLocaleDateString('en', { month: 'short', day: 'numeric' })] = 0;
      }
      items.forEach(item => {
        const ts = item.timestamp || item.id;
        const parsedTs = isNaN(Number(ts)) ? ts : Number(ts);
        const d = new Date(parsedTs).toLocaleDateString('en', { month: 'short', day: 'numeric' });
        if (d in days) days[d]++;
      });
      labels = Object.keys(days); values = Object.values(days);
    }
    const def = this.chartDefaults();
    this.charts.line = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets: [{ label: 'Scans', data: values, borderColor: 'rgba(16,185,129,1)', backgroundColor: 'rgba(16,185,129,0.1)', borderWidth: 2, tension: 0.4, fill: true, pointBackgroundColor: 'rgba(16,185,129,1)', pointRadius: 3 }] },
      options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
    });
  },

  renderMonthly(monthlyData) {
    this.destroy('monthly');
    const ctx = document.getElementById('chart-monthly');
    if (!ctx || !monthlyData || !monthlyData.length) return;
    const labels   = monthlyData.map(m => m.month);
    const healthy  = monthlyData.map(m => m.healthy);
    const diseased = monthlyData.map(m => m.diseased);
    const def = this.chartDefaults();
    this.charts.monthly = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Healthy', data: healthy, backgroundColor: 'rgba(16,185,129,0.7)', borderColor: 'rgba(16,185,129,1)', borderWidth: 1, borderRadius: 6 },
          { label: 'Diseased', data: diseased, backgroundColor: 'rgba(239,68,68,0.7)', borderColor: 'rgba(239,68,68,1)', borderWidth: 1, borderRadius: 6 }
        ]
      },
      options: { plugins: { legend: { labels: { color: '#94a3b8' } } }, scales: def.scales, responsive: true, maintainAspectRatio: false }
    });
  },

  renderRisk(breakdown, items) {
    this.destroy('risk');
    const ctx = document.getElementById('chart-risk');
    if (!ctx) return;
    let high = 0, medium = 0, low = 0;
    if (breakdown) { high = breakdown.HIGH || 0; medium = breakdown.MEDIUM || 0; low = breakdown.LOW || 0; }
    else { items.forEach(i => { if (i.riskLevel === 'HIGH') high++; else if (i.riskLevel === 'MEDIUM') medium++; else low++; }); }
    this.charts.risk = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['High Risk', 'Medium Risk', 'Low Risk'],
        datasets: [{ data: [high, medium, low],
          backgroundColor: ['rgba(239,68,68,0.7)', 'rgba(245,158,11,0.7)', 'rgba(16,185,129,0.7)'],
          borderColor:     ['rgba(239,68,68,1)',   'rgba(245,158,11,1)',   'rgba(16,185,129,1)'],
          borderWidth: 1, borderRadius: 8 }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
                  y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } } },
        responsive: true, maintainAspectRatio: false
      }
    });
  },

  renderActivity(activity) {
    const list = document.getElementById('activity-list');
    if (!list) return;
    if (!activity || !activity.length) {
      list.innerHTML = '<div class="activity-empty">No recent scans to show.</div>';
      return;
    }
    const fmt = iso => iso ? new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';
    list.innerHTML = activity.map(a => `
      <div class="activity-item">
        <div class="activity-dot ${a.isHealthy ? 'healthy' : 'diseased'}"></div>
        <div class="activity-info">
          <div class="activity-disease">${(a.disease || 'Unknown').split('___').pop().replace(/_/g, ' ')}</div>
          <div class="activity-date">${fmt(a.date)}</div>
        </div>
        ${a.confidence ? `<span style="color:var(--text2);font-size:.72rem;margin-right:4px">${a.confidence}%</span>` : ''}
        <span class="activity-risk ${a.riskLevel || 'LOW'}">${a.riskLevel || 'LOW'}</span>
      </div>`).join('');
  }
};

// ── WEATHER MODULE (Phase 1C + Bug #2 Geolocation Fix) ──────
const Weather = {
  _refreshInterval: null,
  _usingCoords: false,

  // Load by city name
  async load(city) {
    city = (city || '').trim() || (document.getElementById('weather-city-input') || {}).value || 'Bhopal';
    console.log('[Weather] Loading city:', city);
    try {
      const res  = await fetch(`${API_URL}/api/weather?city=${encodeURIComponent(city)}`);
      const data = await res.json();
      this._usingCoords = false;
      this.render(data);
    } catch (e) {
      console.warn('[Weather] City load failed:', e.message);
    }
  },

  // Load by GPS coordinates (Bug #2 fix)
  async loadByCoords(lat, lon) {
    console.log('[Weather] Loading by coords:', lat, lon);
    try {
      const res  = await fetch(`${API_URL}/api/weather?lat=${lat}&lon=${lon}`);
      const data = await res.json();
      this._usingCoords = true;
      // Update city input to show detected city name
      const input = document.getElementById('weather-city-input');
      if (input && data.city && data.city !== 'Your Location') input.value = data.city;
      this.render(data);
    } catch (e) {
      console.warn('[Weather] Coords load failed, falling back to Bhopal:', e.message);
      this.load('Bhopal');
    }
  },

  // Request browser geolocation, fallback to Bhopal
  getCurrentLocationWeather() {
    if (!navigator.geolocation) {
      console.log('[Weather] Geolocation not supported, using Bhopal');
      this.load('Bhopal');
      return;
    }
    console.log('[Weather] Requesting geolocation...');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        console.log('[Weather] GPS acquired:', latitude, longitude);
        this.loadByCoords(latitude, longitude);
      },
      (err) => {
        console.log('[Weather] Geolocation denied/failed:', err.message, '— using Bhopal');
        this.load('Bhopal');
      },
      { timeout: 8000, maximumAge: 300000 }
    );
  },

  render(data) {
    const setEl = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    // Location label
    const locEl = document.getElementById('weather-location');
    if (locEl) {
      locEl.textContent = data.coordBased
        ? (data.city && data.city !== 'Your Location' ? `📍 ${data.city}` : '📍 Current Location')
        : (data.city || 'Unknown');
    }

    setEl('w-temp',      data.temperature !== undefined ? data.temperature + '°C'    : '—');
    setEl('w-humid',     data.humidity    !== undefined ? data.humidity    + '%'      : '—');
    setEl('w-rain',      data.rainChance  !== undefined ? data.rainChance  + '%'      : '—');
    setEl('w-wind',      data.windSpeed   !== undefined ? data.windSpeed   + ' km/h'  : '—');
    setEl('w-condition', data.condition   || '—');
    setEl('weather-updated', (data.coordBased ? '📍 Using Current Location • ' : '') +
      'Updated: ' + new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }));

    const badge = document.getElementById('weather-source-badge');
    if (badge) {
      if (data.coordBased) {
        badge.textContent = data.source === 'live' ? '📍 Live Location' : '📍 Location (Sim)';
        badge.className   = 'weather-source-badge' + (data.source === 'live' ? '' : ' simulated');
      } else {
        badge.textContent = data.source === 'live' ? 'Live' : 'Simulated';
        badge.className   = 'weather-source-badge' + (data.source === 'live' ? '' : ' simulated');
      }
    }

    this.renderInsights(data.insights || []);
  },

  renderInsights(insights) {
    const list = document.getElementById('insights-list');
    if (!list) return;
    if (!insights.length) { list.innerHTML = '<div class="insight-loading">No insights available.</div>'; return; }
    list.innerHTML = insights.map(ins => `
      <div class="insight-item ${ins.level}">
        <span class="insight-icon">${ins.icon || '🌿'}</span>
        <span>${ins.message}</span>
      </div>`).join('');
  },

  init() {
    const btn   = document.getElementById('weather-refresh-btn');
    const input = document.getElementById('weather-city-input');
    if (btn)   btn.addEventListener('click', () => this.load(input?.value));
    if (input) input.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.load(input.value); });

    // Bug #2 fix: Auto-detect location on load
    this.getCurrentLocationWeather();

    // Auto-refresh every 30 min (use coords if we have them, else city)
    this._refreshInterval = setInterval(() => {
      if (this._usingCoords) {
        this.getCurrentLocationWeather();
      } else {
        this.load(input?.value || 'Bhopal');
      }
    }, 30 * 60 * 1000);
  }
};

// ── RISK CARD (Phase 1E) ────────────────────────────────────
const RiskCard = {
  show(riskLevel, riskScore, riskReason, weatherSnap) {
    const card   = document.getElementById('risk-card');
    const badge  = document.getElementById('risk-badge');
    const label  = document.getElementById('risk-score-label');
    const ring   = document.getElementById('risk-ring-fg');
    const reason = document.getElementById('risk-reason');
    const snap   = document.getElementById('risk-weather-snap');
    if (!card) return;

    card.classList.remove('hidden', 'risk-high', 'risk-medium', 'risk-low');
    const rl = (riskLevel || 'LOW').toLowerCase();
    card.classList.add('risk-' + rl);

    if (badge) {
      badge.textContent = riskLevel || 'LOW';
      badge.className   = 'risk-badge ' + rl;
    }

    const score = Math.min(100, Math.max(0, riskScore || 0));
    if (label) label.textContent = score + '%';
    if (ring) {
      const circumference = 201;
      const offset = circumference - (circumference * score / 100);
      ring.style.strokeDashoffset = offset;
      const colors = { high: '#ef4444', medium: '#f59e0b', low: '#10b981' };
      ring.style.stroke = colors[rl] || '#7c3aed';
    }
    if (reason) reason.textContent = riskReason || 'Analysis complete.';

    if (snap && weatherSnap) {
      snap.innerHTML = `
        <span class="risk-snap-pill">🌡️ ${weatherSnap.temperature || '—'}°C</span>
        <span class="risk-snap-pill">💧 ${weatherSnap.humidity || '—'}%</span>
        <span class="risk-snap-pill">🌧️ ${weatherSnap.rainChance || '—'}% rain</span>`;
    }
  },

  hide() {
    const card = document.getElementById('risk-card');
    if (card) card.classList.add('hidden');
  }
};

// ── FEEDBACK (Phase 2B — MongoDB-backed) ────────
const Feedback = {
  selected: 0,
  init() {
    document.querySelectorAll(".star").forEach(s => {
      s.addEventListener("mouseenter", () => this.highlight(+s.dataset.val));
      s.addEventListener("mouseleave", () => this.highlight(this.selected));
      s.addEventListener("click", () => {
        this.selected = +s.dataset.val;
        this.highlight(this.selected);
        const labels = ["","Poor","Fair","Good","Great","Excellent!"];
        const hiLabels = ["","खराब","ठीक","अच्छा","बहुत अच्छा","उत्कृष्ट!"];
        const lbl = I18n.getLang() === 'hi' ? hiLabels : labels;
        document.getElementById("star-label").textContent = lbl[this.selected];
      });
    });
    document.getElementById("feedback-submit-btn").addEventListener("click", () => this.submit());
    document.getElementById("go-scan-btn")?.addEventListener("click", () => Router.navigate("home"));
    document.getElementById("go-scan-analytics-btn")?.addEventListener("click", () => Router.navigate("home"));
  },
  highlight(val) { document.querySelectorAll(".star").forEach(s => s.classList.toggle("active", +s.dataset.val <= val)); },
  async submit() {
    if (!this.selected) { Toast.show(I18n.t('feedback_select_rating','Please select a star rating.'), "error"); return; }
    if (!window.Auth || !Auth.isLoggedIn()) { Toast.show(I18n.t('feedback_login_required','Please login to submit feedback.'), "error"); return; }
    const text = (document.getElementById("feedback-text").value || "").trim();
    if (text.length < 10) { Toast.show(I18n.t('feedback_min_chars','Message must be at least 10 characters.'), "error"); return; }
    if (text.length > 1000) { Toast.show(I18n.t('feedback_max_chars','Message cannot exceed 1000 characters.'), "error"); return; }
    const btn = document.getElementById("feedback-submit-btn");
    btn.disabled = true;
    try {
      const res = await fetch(API_URL + '/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...Auth.getAuthHeaders() },
        body: JSON.stringify({ rating: this.selected, message: text })
      });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Submit failed');
      this.selected = 0; this.highlight(0);
      document.getElementById("feedback-text").value = "";
      document.getElementById("feedback-char-count").textContent = "0 / 1000";
      document.getElementById("star-label").textContent = I18n.t('star_label_default','Click to rate');
      Toast.show(I18n.t('feedback_success','Thank you for your feedback! ⭐'), "success");
      this.render();
    } catch(e) {
      Toast.show('Failed: ' + e.message, "error");
    } finally { btn.disabled = false; }
  },
  async render() {
    // Load public stats
    try {
      const sr = await fetch(API_URL + '/api/feedback/stats');
      const sd = await sr.json();
      if (sd.success) {
        const avg = document.getElementById("avg-rating");
        const tot = document.getElementById("total-feedback");
        if (avg) avg.textContent = sd.avg_rating ? sd.avg_rating + " ⭐" : "—";
        if (tot) tot.textContent = sd.total || 0;
      }
    } catch(_){}
    // Load user's own feedback if logged in
    const list = document.getElementById("feedback-list");
    if (!list) return;
    if (!window.Auth || !Auth.isLoggedIn()) { list.innerHTML = '<div style="color:var(--text2);font-size:.85rem;padding:12px 0">Login to see your feedback.</div>'; return; }
    try {
      const mr = await fetch(API_URL + '/api/feedback/my', { headers: Auth.getAuthHeaders() });
      const md = await mr.json();
      const items = md.feedbacks || [];
      if (!items.length) { list.innerHTML = '<div style="color:var(--text2);font-size:.85rem;padding:12px 0">No feedback submitted yet.</div>'; return; }
      const fmt = iso => iso ? new Date(iso).toLocaleDateString('en-IN', {day:'numeric',month:'short',year:'numeric'}) : '';
      list.innerHTML = items.slice(0,10).map(item => `
        <div class="feedback-item">
          <div class="feedback-item-stars">${"★".repeat(item.rating)}${"☆".repeat(5-item.rating)}</div>
          ${item.message ? `<div class="feedback-item-text">"${item.message}"</div>` : ""}
          <div class="feedback-item-time">${fmt(item.createdAt)}</div>
        </div>`).join("");
    } catch(_){ list.innerHTML = '<div style="color:var(--text2);font-size:.85rem">Could not load feedback.</div>'; }
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
predictButton.addEventListener('click', async () => {
  const imageUrl = urlUploader.value.trim();
  let requestBody;

  // ── Build request body ──────────────────────────────────────
  if (currentFile) {
    const fd = new FormData();
    fd.append('file', currentFile);
    requestBody = fd;
  } else if (imageUrl) {
    requestBody = JSON.stringify({ url: imageUrl });
  } else {
    Toast.show('Please choose an image file or paste a URL.', 'error');
    return;
  }

  // ── CRITICAL FIX: Always inject JWT into /predict headers ──
  // Without this, get_optional_user_id() on the backend returns None
  // and every scan is saved without a userId (into the guest pool).
  const authHeaders = window.Auth ? Auth.getAuthHeaders() : {};
  const requestHeaders = { ...authHeaders };
  if (typeof requestBody === 'string') {
    requestHeaders['Content-Type'] = 'application/json';
  }
  // NOTE: Do NOT set Content-Type for FormData — browser sets it with boundary automatically.

  console.log('[Predict] isLoggedIn:', window.Auth ? Auth.isLoggedIn() : false,
              '| JWT present:', !!authHeaders['Authorization']);

  setLoading(true);
  AILoader.show();
  preventionCard.classList.add('hidden');
  resultsDiv.classList.add('hidden');
  chatContainer.classList.add('hidden');

  try {
    const res = await fetch(API_URL + '/predict', {
      method: 'POST',
      headers: requestHeaders,
      body: requestBody,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Prediction failed');
    }
    const data = await res.json();
    console.log('[Predict] Response | id:', data.id, '| disease:', data.disease_name,
                '| riskLevel:', data.riskLevel);

    AILoader.hide();
    displayPrediction(data);

    chatContext = { ...chatContext, disease: data.disease_name,
      symptoms: data.symptoms, treatment: data.treatment, prevention: data.prevention };
    chatBox.innerHTML = '';
    addMessageToChat('Welcome! I detected **' + data.disease_name + '**.\n\nAsk me about:\n- **Symptoms**\n- **Treatment**\n- **Prevention**\n\n_I am here to help!_', 'bot');

  } catch (err) {
    AILoader.hide();
    Toast.show('Error: ' + err.message, 'error');
    preventionCard.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  const lbl = predictButton.querySelector('.btn-label');
  const sp  = document.getElementById('btn-spinner');
  predictButton.disabled = on;
  if (lbl) lbl.textContent = on ? 'Analyzing...' : 'Analyze Image';
  if (sp)  sp.classList.toggle('hidden', !on);
  predictButton.classList.toggle('loading', on);
}

function displayPrediction(data) {
  predictionOutput.textContent = data.disease_name;
  symptomsOutput.innerHTML   = (data.symptoms  || '').replace(/\n/g, '<br>');
  treatmentOutput.innerHTML  = (data.treatment || '').replace(/\n/g, '<br>');
  preventionOutput.innerHTML = (data.prevention|| '').replace(/\n/g, '<br>');

  const isHealthy  = data.prediction.includes('healthy');
  const confidence = typeof data.confidence === 'number' ? data.confidence : (isHealthy ? 97 : 92);
  const sev        = Severity.get(confidence);

  const bar   = document.getElementById('confidence-bar');
  const val   = document.getElementById('confidence-value');
  const badge = document.getElementById('severity-badge');
  const warn  = document.getElementById('severity-warning');

  if (bar) setTimeout(() => { bar.style.width = confidence + '%'; }, 300);
  if (val) val.textContent = confidence + '%';

  if (isHealthy) {
    predictionOutput.classList.add('healthy');
    symptomsSection.classList.add('hidden');
    treatmentSection.classList.add('hidden');
    preventionTitle.textContent = 'How to Keep it Healthy';
    if (badge) { badge.textContent = 'Healthy'; badge.className = 'severity-badge sev-low'; }
    if (warn)  warn.classList.add('hidden');
  } else {
    predictionOutput.classList.remove('healthy');
    symptomsSection.classList.remove('hidden');
    treatmentSection.classList.remove('hidden');
    preventionTitle.textContent = 'Prevention';
    if (badge) { badge.textContent = sev.label; badge.className = 'severity-badge ' + sev.cls; }
    if (warn)  warn.classList.toggle('hidden', sev.cls !== 'sev-high');
  }

  resultsDiv.classList.remove('hidden');
  chatContainer.classList.remove('hidden');
  resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // ── Build lastResult for PDF generation ────────────────────
  const imgEl = document.querySelector('#image-preview img');
  const imageDataUrl = (imgEl && imgEl.src && imgEl.src.startsWith('data:')) ? imgEl.src : null;
  lastResult = {
    id:           data.id,
    imageDataUrl,
    disease:      data.disease_name,
    confidence,
    severity:     isHealthy ? 'LOW' : (sev.cls === 'sev-high' ? 'HIGH' : sev.cls === 'sev-medium' ? 'MEDIUM' : 'LOW'),
    isHealthy,
    riskLevel:    data.riskLevel,
    riskScore:    data.riskScore,
    timestamp:    new Date().toISOString(),
  };

  // ── CRITICAL FIX: Do NOT call Storage.addScan() (localStorage) ─────────
  // The scan was already persisted in MongoDB by the /predict backend route.
  // Calling addScan() here was the reason scans appeared in guest mode after logout
  // (localStorage had no userId concept). The server is now the single source of truth.
  // We still update localStorage cache so offline mode and PDF still work.
  Storage.addLocalScan(lastResult);

  const scanLabel = isHealthy ? 'Healthy scan' : data.disease_name;
  Toast.show('Scan saved: ' + scanLabel + ' (' + confidence + '%)', 'success');

  // Show risk card
  if (data.riskLevel) {
    RiskCard.show(data.riskLevel, data.riskScore, data.riskReason, null);
  } else {
    RiskCard.hide();
  }
}
document.getElementById('download-pdf-btn') && document.getElementById('download-pdf-btn').addEventListener('click', () => PDF.generate());
document.getElementById('save-history-manual-btn') && document.getElementById('save-history-manual-btn').addEventListener('click', () => { if (lastResult) { Storage.addScan(lastResult); Toast.show('Saved!', 'success'); } else Toast.show('No result yet.', 'info'); });
document.getElementById('clear-all-btn') && document.getElementById('clear-all-btn').addEventListener('click', () => HistoryManager.clearAll());
chatSendButton.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keyup', (e) => { if (e.key === 'Enter') sendChatMessage(); });
async function sendChatMessage() { const message = chatInput.value.trim(); if (!message) return; addMessageToChat(message, 'user'); chatInput.value = ''; chatSendButton.disabled = true; const typing = createTypingIndicator(); chatBox.appendChild(typing); chatBox.scrollTop = chatBox.scrollHeight; try { const res = await fetch(API_URL + '/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, session_id: chatContext.session_id, disease: chatContext.disease, symptoms: chatContext.symptoms, treatment: chatContext.treatment, prevention: chatContext.prevention, language: I18n.getLang() }) }); if (chatBox.contains(typing)) chatBox.removeChild(typing); if (!res.ok) throw new Error('Chat failed'); const data = await res.json(); const isLocal = data.mode === 'local'; if (isLocal) showLocalModeBadge(); addMessageToChat(data.response, 'bot', isLocal); } catch { if (chatBox.contains(typing)) chatBox.removeChild(typing); showLocalModeBadge(); addMessageToChat('Using offline mode. Ask me about symptoms, treatment or prevention!', 'bot', true); } finally { chatSendButton.disabled = false; chatInput.focus(); } }
function showLocalModeBadge() { const b = document.getElementById('local-mode-badge'); if (b) b.classList.remove('hidden'); }
function createTypingIndicator() { const w = document.createElement('div'); w.classList.add('chat-msg','bot','typing-indicator'); w.innerHTML = '<span class=typing-label>Expert is thinking</span><span class=dots><span></span><span></span><span></span></span>'; return w; }
function addMessageToChat(text, sender, isLocalMode = false) { const div = document.createElement('div'); div.classList.add('chat-msg', sender); if (sender === 'bot' && isLocalMode) div.classList.add('local-mode'); let html = text.replace(/\*\*(.*?)\*\*/g,'<strong></strong>').replace(/_(.*?)_/g,'<em></em>').replace(/^[-]\s(.+)/gm,'<li></li>').replace(/(<li>.*<\/li>\n?)+/g,m=>'<ul>'+m+'</ul>').replace(/\n/g,'<br>'); if (sender === 'bot' && isLocalMode) html = '<div class=local-badge>Local Expert Mode</div>' + html; div.innerHTML = html; chatBox.appendChild(div); chatBox.scrollTop = chatBox.scrollHeight; }
// ── PROFILE PAGE ───────────────────────────────
const ProfilePage = {
  async render() {
    if (!window.Auth || !Auth.isLoggedIn()) {
      // Redirect to home if not logged in
      Router.navigate('home');
      if (window.Auth) Auth.showLogin();
      return;
    }
    const user = await Auth.fetchProfile();
    if (!user) return;
    const fmt = (iso) => iso ? new Date(iso).toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }) : '—';
    const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setEl('profile-name',        user.name  || '—');
    setEl('profile-email',       user.email || '—');
    setEl('profile-total-scans', user.totalScans ?? 0);
    setEl('profile-join-date',   fmt(user.createdAt));
    setEl('profile-last-login',  fmt(user.lastLogin));
    setEl('profile-account-type', (user.role || 'user').charAt(0).toUpperCase() + (user.role || 'user').slice(1));
    setEl('profile-role-badge',  (user.role || 'User').toUpperCase());
    const nameInput = document.getElementById('profile-edit-name');
    if (nameInput) nameInput.value = user.name || '';
    // Profile update form
    const form = document.getElementById('form-profile-update');
    if (form && !form._bound) {
      form._bound = true;
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('btn-profile-save');
        const errEl = document.getElementById('profile-edit-error');
        const newName = (document.getElementById('profile-edit-name')?.value || '').trim();
        if (!newName) { errEl.textContent = 'Name cannot be empty.'; errEl.classList.remove('hidden'); return; }
        btn.disabled = true; btn.textContent = 'Saving…';
        const res = await Auth.updateProfile({ name: newName });
        btn.disabled = false; btn.textContent = '💾 Save Changes';
        if (res.success) {
          // Update local user cache
          const u = Auth.getUser(); if (u) { u.name = newName; Auth.login(Auth.getToken(), u); }
          Toast.show('✅ Profile updated!', 'success');
          ProfilePage.render();
        } else {
          errEl.textContent = res.message || 'Update failed.'; errEl.classList.remove('hidden');
        }
      });
    }
  }
};

Voice.init(); Camera.init(); Feedback.init(); Weather.init();
// Phase 2A: init language system after DOM ready
I18n.init();
// Phase 2B: init char counter
(function(){
  const ta = document.getElementById('feedback-text');
  const cc = document.getElementById('feedback-char-count');
  if (ta && cc) {
    ta.addEventListener('input', () => {
      const len = ta.value.length;
      cc.textContent = len + ' / 1000';
      cc.className = 'feedback-char-count' + (len > 1000 ? ' over' : len > 900 ? ' warn' : '');
    });
  }
})();

// ── Auth Init — guaranteed after full DOM parse ────────────
if (typeof window.Auth !== 'undefined') {
  Auth.init();
} else {
  document.addEventListener('DOMContentLoaded', function() {
    if (typeof window.Auth !== 'undefined') Auth.init();
  });
}
window.AgroAI = { PDF, Router, Toast, HistoryManager, Analytics, Feedback, Storage, ProfilePage, Weather, RiskCard, I18n };

// Re-render data pages on auth changes (login/logout)
document.addEventListener('agroai:auth', (e) => {
  console.log('[agroai:auth] Auth event fired:', e.detail.type);
  if (e.detail.type === 'logout') {
    // Clear ALL local caches so next guest session is clean
    Storage.clearLocalHistory();
    console.log('[agroai:auth] Local history cleared on logout');
  }
  // Always re-render whatever page is active, so data refreshes immediately
  const page = Router.current;
  console.log('[agroai:auth] Re-rendering page:', page);
  if (page === 'history')   HistoryManager.render();
  if (page === 'analytics') Analytics.render();
  if (page === 'profile')   ProfilePage.render();
});

// =============================================================================
// PHASE 3A — RAG Agriculture Expert Chat Module
// =============================================================================
const RAGChat = (() => {
  let _busy = false;

  // ── DOM refs (resolved lazily after DOMContentLoaded) ──────────────────
  const $ = id => document.getElementById(id);

  // ── Markdown-lite renderer (bold, bullet lists) ────────────────────────
  function _renderMarkdown(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
      .replace(/\n{2,}/g, '<br><br>')
      .replace(/\n/g, '<br>');
  }

  // ── Add a chat bubble ──────────────────────────────────────────────────
  function _addMsg(html, role, extraClass = '') {
    const box = $('rag-chat-box');
    if (!box) return;
    const div = document.createElement('div');
    div.className = `rag-msg ${role}${extraClass ? ' ' + extraClass : ''}`;
    div.innerHTML = html;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  // ── Typing indicator ───────────────────────────────────────────────────
  function _showTyping() {
    const box = $('rag-chat-box');
    if (!box) return null;
    const el = document.createElement('div');
    el.className = 'typing-indicator';
    el.id = 'rag-typing';
    el.innerHTML = `
      <span class="typing-label">Agriculture Expert is searching knowledge base…</span>
      <div class="dots"><span></span><span></span><span></span></div>`;
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
    return el;
  }

  // ── Render source citation pills ──────────────────────────────────────
  function _renderSources(sources) {
    const panel = $('rag-sources-panel');
    const list  = $('rag-sources-list');
    if (!panel || !list) return;

    if (!sources || sources.length === 0) {
      panel.classList.add('hidden');
      return;
    }

    list.innerHTML = sources.map(s => {
      const doc  = (s.document  || 'Unknown').replace(/_/g, ' ').replace('.txt','').replace('.pdf','');
      const page = s.page  || 1;
      const cat  = (s.category || 'general');
      return `
        <span class="rag-source-pill" title="${s.document}">
          <span class="pill-icon">📄</span>
          <span class="pill-doc">${doc}</span>
          <span class="pill-page">p.${page}</span>
          <span class="pill-cat">[${cat}]</span>
        </span>`;
    }).join('');

    panel.classList.remove('hidden');
  }

  // ── Update the mode badge in the header ───────────────────────────────
  function _updateBadge(mode) {
    const badge = $('rag-mode-badge');
    if (!badge) return;
    badge.className = 'rag-mode-badge';
    if (mode === 'rag') {
      badge.classList.add('mode-rag');
      badge.textContent = '🔬 Knowledge Base';
    } else if (mode === 'fallback') {
      badge.classList.add('mode-fallback');
      badge.textContent = '⚠️ General AI';
    } else {
      badge.classList.add('mode-ready');
      badge.textContent = '🔬 Ready';
    }
  }

  // ── Main send function ─────────────────────────────────────────────────
  async function send(question) {
    question = (question || '').trim();
    if (!question || _busy) return;

    _busy = true;
    const sendBtn = $('rag-send-btn');
    const input   = $('rag-question-input');
    if (sendBtn) sendBtn.disabled = true;

    // Add user bubble
    _addMsg(question, 'user');
    if (input) input.value = '';

    // Typing indicator
    const typing = _showTyping();

    try {
      // Detect language from I18n module (falls back to 'en')
      const lang = (typeof I18n !== 'undefined' && I18n.getLang) ? I18n.getLang() : 'en';

      const res = await fetch(`${API_URL}/api/rag-chat`, {
        method:  'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(Auth && Auth.getAuthHeaders ? Auth.getAuthHeaders() : {}),
        },
        body: JSON.stringify({ question, language: lang }),
      });

      if (typing) typing.remove();

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        _addMsg(`❌ ${err.error || 'Request failed (' + res.status + ')'}`, 'bot');
        return;
      }

      const data = await res.json();
      const mode    = data.mode    || 'fallback';
      const answer  = data.answer  || '';
      const sources = data.sources || [];

      const extraClass = mode === 'fallback' ? 'fallback-msg' : '';
      _addMsg(_renderMarkdown(answer), 'bot', extraClass);
      _renderSources(sources);
      _updateBadge(mode);

    } catch (err) {
      if (typing) typing.remove();
      _addMsg('🌐 Could not reach the server. Please check your connection and try again.', 'bot');
      console.error('[RAGChat] fetch error:', err);
    } finally {
      _busy = false;
      if (sendBtn) sendBtn.disabled = false;
    }
  }

  // ── Voice input (reuses existing SpeechRecognition pattern) ───────────
  function _initVoice() {
    const voiceBtn = $('rag-voice-btn');
    const input    = $('rag-question-input');
    if (!voiceBtn || !input) return;

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { voiceBtn.style.display = 'none'; return; }

    const recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-IN';

    recognition.onresult = e => {
      input.value = e.results[0][0].transcript;
      voiceBtn.classList.remove('listening');
    };
    recognition.onerror = () => voiceBtn.classList.remove('listening');
    recognition.onend   = () => voiceBtn.classList.remove('listening');

    voiceBtn.addEventListener('click', () => {
      voiceBtn.classList.add('listening');
      recognition.start();
    });
  }

  // ── Welcome message ────────────────────────────────────────────────────
  function _welcome() {
    _addMsg(
      '👋 <strong>Namaste! I am the AgroAI Agriculture Expert.</strong><br><br>' +
      'My answers are grounded in verified ICAR agricultural documents. I can help with:<br><br>' +
      '<ul>' +
      '<li>🍅 <strong>Plant Diseases</strong> — identification & chemical/biological control</li>' +
      '<li>🌱 <strong>Fertilizers</strong> — NPK recommendations by crop</li>' +
      '<li>💧 <strong>Irrigation</strong> — drip system setup & fertigation</li>' +
      '<li>🏛️ <strong>Government Schemes</strong> — PM-KISAN, PMFBY, KCC</li>' +
      '<li>🌾 <strong>Crop Management</strong> — sowing, varieties, pest control</li>' +
      '<li>🌧️ <strong>Weather Advisories</strong> — monsoon & frost protection</li>' +
      '</ul><br>' +
      'Use the quick buttons below or type your question!',
      'bot'
    );
  }

  // ── Public init ────────────────────────────────────────────────────────
  function init() {
    const sendBtn = $('rag-send-btn');
    const input   = $('rag-question-input');
    if (!sendBtn || !input) return; // RAG panel not in DOM

    // Send on button click
    sendBtn.addEventListener('click', () => send(input.value));

    // Send on Enter key
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input.value); }
    });

    // Quick question buttons
    document.querySelectorAll('.rag-quick-btn').forEach(btn => {
      btn.addEventListener('click', () => send(btn.dataset.q));
    });

    // Voice
    _initVoice();

    // Welcome message
    _welcome();
  }

  return { init, send };
})();

// ── Auto-init RAGChat on DOMContentLoaded ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  RAGChat.init();
});
