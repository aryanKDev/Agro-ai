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

// ═══════════════════════════════════════════════════════════════════════
// PHASE 4A — ADMIN INTELLIGENCE DASHBOARD
// ═══════════════════════════════════════════════════════════════════════
const AdminDashboard = (() => {
  'use strict';

  const _charts = {};
  let _loadedTabs = new Set();

  // ── Chart.js defaults for admin theme ─────────────────────────────
  function _def() {
    return {
      plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } } },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    };
  }

  function _destroyChart(id) {
    if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
  }

  function _setEl(id, v) {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  }

  function _authHeaders() {
    return window.Auth ? Auth.getAuthHeaders() : {};
  }

  // ── Tab switching ─────────────────────────────────────────────────
  function _initTabs() {
    document.querySelectorAll('.admin-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        // Update active button
        document.querySelectorAll('.admin-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        // Show correct panel
        document.querySelectorAll('.admin-tab-content').forEach(p => p.classList.remove('active'));
        const panel = document.getElementById('admin-panel-' + tab);
        if (panel) panel.classList.add('active');
        // Lazy-load tab data
        if (!_loadedTabs.has(tab)) {
          _loadTab(tab);
        }
      });
    });
  }

  async function _loadTab(tab) {
    _loadedTabs.add(tab);
    const endpointMap = {
      overview:    '/api/admin/overview',
      agriculture: '/api/admin/agriculture',
      rag:         '/api/admin/rag',
      feedback:    '/api/admin/feedback',
      languages:   '/api/admin/languages',
    };
    const url = endpointMap[tab];
    if (!url) return;
    try {
      const res  = await fetch(API_URL + url, { headers: _authHeaders() });
      if (!res.ok) {
        if (res.status === 403) { _showAccessDenied(); return; }
        throw new Error('HTTP ' + res.status);
      }
      const data = await res.json();
      if (tab === 'overview')    _renderOverview(data);
      if (tab === 'agriculture') _renderAgriculture(data);
      if (tab === 'rag')         _renderRAG(data);
      if (tab === 'feedback')    _renderFeedback(data);
      if (tab === 'languages')   _renderLanguages(data);
    } catch (e) {
      console.error('[AdminDashboard] loadTab error:', tab, e);
      if (window.Toast) Toast.show('Admin data load failed: ' + e.message, 'error');
    }
  }

  // ── Show/hide access denied ────────────────────────────────────────
  function _showAccessDenied() {
    const denied  = document.getElementById('admin-access-denied');
    const content = document.getElementById('admin-content');
    if (denied)  denied.classList.remove('hidden');
    if (content) content.classList.add('hidden');
  }

  function _showContent() {
    const denied  = document.getElementById('admin-access-denied');
    const content = document.getElementById('admin-content');
    if (denied)  denied.classList.add('hidden');
    if (content) content.classList.remove('hidden');
  }

  // ── OVERVIEW TAB ─────────────────────────────────────────────────
  function _renderOverview(data) {
    _setEl('adm-total-users',    data.totalUsers   ?? '—');
    _setEl('adm-active-users',   data.activeUsers  ?? '—');
    _setEl('adm-total-scans',    data.totalScans   ?? '—');
    _setEl('adm-total-feedback', data.totalFeedback ?? '—');
    _setEl('adm-total-rag',      data.totalRagQueries ?? '—');
    _setEl('adm-avg-scans',      data.avgScansPerUser ?? '—');
    // 30-day growth chart uses scan data from agriculture tab — placeholder trend
    // We'll draw a simple line using total scan count (single data point)
    // A real trend needs per-day data; load it from the RAG trend if available
    _destroyChart('growth');
    const ctx = document.getElementById('adm-chart-growth');
    if (ctx) {
      // Generate simulated daily labels for demo when no trend data available
      const labels = [], values = [];
      const now = new Date();
      for (let i = 29; i >= 0; i--) {
        const d = new Date(now - i * 86400000);
        labels.push(d.toLocaleDateString('en', { month: 'short', day: 'numeric' }));
        values.push(0);
      }
      const def = _def();
      _charts.growth = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Daily Scans',
            data: values,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: '#f59e0b',
            pointRadius: 3,
          }]
        },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
      });
    }
  }

  // ── AGRICULTURE TAB ───────────────────────────────────────────────
  function _renderAgriculture(data) {
    _setEl('adm-high-risk',   data.highRiskCount  ?? '—');
    _setEl('adm-avg-risk',    (data.avgRiskScore ?? '—') + (data.avgRiskScore ? '%' : ''));
    _setEl('adm-sev-high',    data.severityBreakdown?.HIGH   ?? '—');
    _setEl('adm-sev-medium',  data.severityBreakdown?.MEDIUM ?? '—');

    const def = _def();

    // Top diseases bar chart
    _destroyChart('diseases');
    const dCtx = document.getElementById('adm-chart-diseases');
    if (dCtx && data.topDiseases?.length) {
      const labels = data.topDiseases.slice(0, 8).map(d => (d.disease || '').split('___').pop().replace(/_/g,' ').slice(0,18));
      const values = data.topDiseases.slice(0, 8).map(d => d.count);
      _charts.diseases = new Chart(dCtx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Cases', data: values, backgroundColor: 'rgba(245,158,11,0.7)', borderColor: 'rgba(245,158,11,1)', borderWidth: 1, borderRadius: 6 }] },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
      });
    }

    // Most affected crops horizontal bar
    _destroyChart('crops');
    const cCtx = document.getElementById('adm-chart-crops');
    if (cCtx && data.mostCommonCropIssues?.length) {
      const labels = data.mostCommonCropIssues.map(c => c.crop);
      const values = data.mostCommonCropIssues.map(c => c.count);
      _charts.crops = new Chart(cCtx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Issues', data: values, backgroundColor: 'rgba(16,185,129,0.7)', borderColor: 'rgba(16,185,129,1)', borderWidth: 1, borderRadius: 6 }] },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
      });
    }

    // Severity doughnut
    _destroyChart('severity');
    const sCtx = document.getElementById('adm-chart-severity');
    if (sCtx && data.severityBreakdown) {
      const sb = data.severityBreakdown;
      _charts.severity = new Chart(sCtx, {
        type: 'doughnut',
        data: {
          labels: ['HIGH', 'MEDIUM', 'LOW'],
          datasets: [{ data: [sb.HIGH || 0, sb.MEDIUM || 0, sb.LOW || 0], backgroundColor: ['rgba(239,68,68,0.75)', 'rgba(245,158,11,0.75)', 'rgba(16,185,129,0.75)'], borderColor: ['rgba(239,68,68,1)', 'rgba(245,158,11,1)', 'rgba(16,185,129,1)'], borderWidth: 2 }]
        },
        options: { plugins: { legend: { labels: { color: '#94a3b8' } } }, cutout: '60%', responsive: true, maintainAspectRatio: false }
      });
    }

    // Weather impact radar/bar
    _destroyChart('weather');
    const wCtx = document.getElementById('adm-chart-weather');
    if (wCtx && data.weatherImpactSummary?.count) {
      const ws = data.weatherImpactSummary;
      _charts.weather = new Chart(wCtx, {
        type: 'bar',
        data: {
          labels: ['Avg Temp (°C)', 'Avg Humidity (%)', 'Avg Rain (%)'],
          datasets: [{ label: 'High-Risk Conditions', data: [ws.avgTemp || 0, ws.avgHumidity || 0, ws.avgRain || 0], backgroundColor: ['rgba(239,68,68,0.6)', 'rgba(59,130,246,0.6)', 'rgba(16,185,129,0.6)'], borderColor: ['rgba(239,68,68,1)', 'rgba(59,130,246,1)', 'rgba(16,185,129,1)'], borderWidth: 1, borderRadius: 8 }]
        },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
      });
    }
  }

  // ── RAG TAB ────────────────────────────────────────────────────────
  function _renderRAG(data) {
    _setEl('adm-rag-total',    data.totalRagQueries  ?? '—');
    _setEl('adm-rag-success',  (data.ragSuccessRate  ?? '—') + (data.ragSuccessRate != null ? '%' : ''));
    _setEl('adm-rag-fallback', (data.fallbackRate    ?? '—') + (data.fallbackRate   != null ? '%' : ''));
    _setEl('adm-rag-sources',  data.topSources?.length ?? '—');

    const def = _def();

    // Chat volume trend line chart
    _destroyChart('ragTrend');
    const tCtx = document.getElementById('adm-chart-rag-trend');
    if (tCtx && data.chatVolumeTrend?.length) {
      const labels = data.chatVolumeTrend.map(d => d.date);
      const values = data.chatVolumeTrend.map(d => d.count);
      _charts.ragTrend = new Chart(tCtx, {
        type: 'line',
        data: { labels, datasets: [{ label: 'RAG Queries', data: values, borderColor: 'rgba(16,185,129,1)', backgroundColor: 'rgba(16,185,129,0.1)', borderWidth: 2, tension: 0.4, fill: true, pointBackgroundColor: 'rgba(16,185,129,1)', pointRadius: 3 }] },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
      });
    }

    // Category distribution doughnut
    _destroyChart('ragCats');
    const cCtx = document.getElementById('adm-chart-rag-categories');
    if (cCtx && data.categoryDistribution?.length) {
      const labels = data.categoryDistribution.map(c => c.category);
      const values = data.categoryDistribution.map(c => c.count);
      const colors = ['rgba(16,185,129,.7)','rgba(124,58,237,.7)','rgba(59,130,246,.7)','rgba(245,158,11,.7)','rgba(239,68,68,.7)','rgba(251,191,36,.7)','rgba(167,139,250,.7)','rgba(52,211,153,.7)'];
      _charts.ragCats = new Chart(cCtx, {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors.slice(0, labels.length), borderWidth: 2, borderColor: 'rgba(15,15,26,1)' }] },
        options: { plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } }, cutout: '55%', responsive: true, maintainAspectRatio: false }
      });
    }

    // RAG vs Fallback pie
    _destroyChart('ragMode');
    const mCtx = document.getElementById('adm-chart-rag-mode');
    if (mCtx) {
      const ragCount = Math.round((data.ragSuccessRate || 0) / 100 * (data.totalRagQueries || 0));
      const fbkCount = (data.totalRagQueries || 0) - ragCount;
      _charts.ragMode = new Chart(mCtx, {
        type: 'doughnut',
        data: { labels: ['RAG Success', 'Fallback'], datasets: [{ data: [ragCount, fbkCount], backgroundColor: ['rgba(16,185,129,0.75)', 'rgba(245,158,11,0.75)'], borderColor: ['rgba(16,185,129,1)', 'rgba(245,158,11,1)'], borderWidth: 2 }] },
        options: { plugins: { legend: { labels: { color: '#94a3b8' } } }, cutout: '60%', responsive: true, maintainAspectRatio: false }
      });
    }

    // Top questions table
    const tbody = document.getElementById('adm-questions-tbody');
    if (tbody && data.topQuestions?.length) {
      tbody.innerHTML = data.topQuestions.map((q, i) => `
        <tr>
          <td class="rank-num">${i + 1}</td>
          <td>${q.question}</td>
          <td><span class="admin-count-badge">${q.count}</span></td>
        </tr>`).join('');
    } else if (tbody) {
      tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--text2)">No RAG queries yet</td></tr>';
    }
  }

  // ── FEEDBACK TAB ──────────────────────────────────────────────────
  function _renderFeedback(data) {
    const avg  = data.avgRating;
    _setEl('adm-fb-avg',   avg != null ? '⭐ ' + avg : '—');
    _setEl('adm-fb-total', data.totalFeedback ?? '—');
    _setEl('adm-fb-5star', data.ratingDistribution?.[5] ?? '—');
    const low = (data.ratingDistribution?.[1] || 0) + (data.ratingDistribution?.[2] || 0);
    _setEl('adm-fb-low', low || '—');

    const def = _def();

    // Rating distribution bars
    const distEl = document.getElementById('adm-rating-dist');
    if (distEl && data.ratingDistribution) {
      const total = data.totalFeedback || 1;
      distEl.innerHTML = [5,4,3,2,1].map(star => {
        const count = data.ratingDistribution[star] || 0;
        const pct   = Math.round(count / total * 100);
        return `<div class="rating-dist-row">
          <span class="rating-dist-label">${'★'.repeat(star)}</span>
          <div class="rating-dist-bar-track"><div class="rating-dist-bar" style="width:${pct}%"></div></div>
          <span class="rating-dist-count">${count}</span>
        </div>`;
      }).join('');
    }

    // Feedback trend line
    _destroyChart('fbTrend');
    const tCtx = document.getElementById('adm-chart-fb-trend');
    if (tCtx && data.feedbackTrend?.length) {
      const labels = data.feedbackTrend.map(d => d.date);
      const values = data.feedbackTrend.map(d => d.count);
      _charts.fbTrend = new Chart(tCtx, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Feedback', data: values, borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.1)', borderWidth: 2, tension: 0.4, fill: true, pointBackgroundColor: '#fbbf24', pointRadius: 3 }] },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false }
      });
    }

    // Keyword cloud
    const cloudEl = document.getElementById('adm-keyword-cloud');
    if (cloudEl && data.keywordFrequency?.length) {
      const maxCount = data.keywordFrequency[0]?.count || 1;
      cloudEl.innerHTML = data.keywordFrequency.map(kw => {
        const ratio = kw.count / maxCount;
        const cls   = ratio > 0.7 ? 'large' : ratio > 0.4 ? 'medium' : '';
        return `<span class="admin-keyword-tag ${cls}">${kw.word} (${kw.count})</span>`;
      }).join('');
    } else if (cloudEl) {
      cloudEl.innerHTML = '<span style="color:var(--text2);font-size:.85rem">No feedback data yet</span>';
    }

    // Latest feedback
    const listEl = document.getElementById('adm-feedback-list');
    if (listEl && data.latestFeedback?.length) {
      const fmt = iso => iso ? new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—';
      listEl.innerHTML = data.latestFeedback.map(f => `
        <div class="admin-feedback-item">
          <div class="admin-feedback-stars">${'★'.repeat(f.rating || 0)}${'☆'.repeat(5 - (f.rating || 0))}</div>
          <div class="admin-feedback-name">${f.name || 'Anonymous'}</div>
          <div class="admin-feedback-text">${f.message || ''}</div>
          <div class="admin-feedback-date">${fmt(f.date)}</div>
        </div>`).join('');
    } else if (listEl) {
      listEl.innerHTML = '<div style="color:var(--text2);font-size:.85rem;padding:20px 0;text-align:center">No feedback yet</div>';
    }
  }

  // ── LANGUAGES TAB ─────────────────────────────────────────────────
  function _renderLanguages(data) {
    _setEl('adm-lang-en-count', data.englishCount    ?? '—');
    _setEl('adm-lang-hi-count', data.hindiCount      ?? '—');
    _setEl('adm-lang-top',      data.mostUsedLanguage ?? '—');
    _setEl('adm-lang-total',    (data.englishCount || 0) + (data.hindiCount || 0));

    // Usage bars
    const enBar  = document.getElementById('adm-lang-en-bar');
    const hiBar  = document.getElementById('adm-lang-hi-bar');
    const enPct  = document.getElementById('adm-lang-en-pct');
    const hiPct  = document.getElementById('adm-lang-hi-pct');
    setTimeout(() => {
      if (enBar) enBar.style.width = (data.englishPct || 0) + '%';
      if (hiBar) hiBar.style.width = (data.hindiPct  || 0) + '%';
    }, 100);
    if (enPct) enPct.textContent = (data.englishPct || 0) + '%';
    if (hiPct) hiPct.textContent = (data.hindiPct  || 0) + '%';

    const def = _def();

    // Language pie
    _destroyChart('langPie');
    const pCtx = document.getElementById('adm-chart-lang-pie');
    if (pCtx) {
      _charts.langPie = new Chart(pCtx, {
        type: 'doughnut',
        data: {
          labels: ['English', 'Hindi'],
          datasets: [{ data: [data.englishCount || 0, data.hindiCount || 0], backgroundColor: ['rgba(124,58,237,0.75)', 'rgba(16,185,129,0.75)'], borderColor: ['rgba(124,58,237,1)', 'rgba(16,185,129,1)'], borderWidth: 2 }]
        },
        options: { plugins: { legend: { labels: { color: '#94a3b8' } } }, cutout: '60%', responsive: true, maintainAspectRatio: false }
      });
    }

    // Language trend stacked line
    _destroyChart('langTrend');
    const lCtx = document.getElementById('adm-chart-lang-trend');
    if (lCtx && data.languageTrend?.length) {
      const labels = data.languageTrend.map(d => d.date);
      const enData = data.languageTrend.map(d => d.english);
      const hiData = data.languageTrend.map(d => d.hindi);
      _charts.langTrend = new Chart(lCtx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: 'English', data: enData, borderColor: 'rgba(124,58,237,1)', backgroundColor: 'rgba(124,58,237,0.12)', borderWidth: 2, tension: 0.4, fill: true, pointRadius: 2 },
            { label: 'Hindi',   data: hiData, borderColor: 'rgba(16,185,129,1)',  backgroundColor: 'rgba(16,185,129,0.12)',  borderWidth: 2, tension: 0.4, fill: true, pointRadius: 2 }
          ]
        },
        options: { plugins: { legend: { labels: { color: '#94a3b8' } } }, scales: def.scales, responsive: true, maintainAspectRatio: false }
      });
    }
  }

  // ── Public: render (called by Router when page == 'admin') ─────────
  async function render() {
    // Gate: must be logged in
    if (!window.Auth || !Auth.isLoggedIn()) {
      _showAccessDenied();
      return;
    }

    // Check role from stored user object first (fast)
    const user = Auth.getUser();
    if (user && user.role !== 'admin') {
      _showAccessDenied();
      return;
    }

    _showContent();
    _initTabs();

    // Automatically load overview tab on first render
    if (!_loadedTabs.has('overview')) {
      await _loadTab('overview');
    }
  }

  // ── Public: reset (called on logout) ─────────────────────────────
  function reset() {
    _loadedTabs.clear();
    Object.keys(_charts).forEach(k => { if (_charts[k]) { _charts[k].destroy(); delete _charts[k]; } });
  }

  return { render, reset };
})();

// ── Extend Router to include admin page ──────────────────────────────────────
(function _patchRouter() {
  const orig = Router.navigate.bind(Router);
  Router.pages.push('admin');
  Router.pages.push('plants');   // Phase 4B — add here so page toggle includes it
  Router.navigate = function(page) {
    this.pages.forEach(p => {
      const el = document.getElementById('page-' + p);
      if (el) el.classList.toggle('hidden', p !== page);
    });
    document.querySelectorAll('.nav-page-link').forEach(a => {
      a.classList.toggle('active-nav', a.dataset.page === page);
    });
    const navLinks = document.getElementById('nav-links');
    if (navLinks) navLinks.classList.remove('open');
    this.current = page;
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (page === 'history')   HistoryManager.render();
    if (page === 'analytics') Analytics.render();
    if (page === 'feedback')  Feedback.render();
    if (page === 'profile')   ProfilePage.render();
    if (page === 'admin')     AdminDashboard.render();
    if (page === 'plants')    PlantTracker.render();   // Phase 4B
  };
})();

// ── Wire admin nav badge ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const btnAdmin = document.getElementById('btn-nav-admin');
  if (btnAdmin) {
    btnAdmin.addEventListener('click', e => {
      e.preventDefault();
      Router.navigate('admin');
    });
  }
});

// ── On logout: reset admin dashboard state ───────────────────────────────────
document.addEventListener('agroai:auth', e => {
  if (e.detail?.type === 'logout') {
    AdminDashboard.reset();
    PlantTracker.reset();
  }
});

// ════════════════════════════════════════════════════════════════════════════
// PHASE 4B — PlantTracker Module
// ════════════════════════════════════════════════════════════════════════════
const PlantTracker = (() => {
  'use strict';

  // ── State ─────────────────────────────────────────────────────────────────
  let _plants        = [];
  let _selectedId    = null;
  let _charts        = {};
  let _lastScanData  = null;   // holds current predict result for tracking

  // ── Helpers ───────────────────────────────────────────────────────────────
  function _token() {
    // auth.js stores JWT under 'agroai_jwt' — match that key exactly
    return window.Auth?.getToken?.() || localStorage.getItem('agroai_jwt') || '';
  }

  function _authHeaders() {
    // Reuse Auth.getAuthHeaders() if available (returns {} when logged out)
    if (window.Auth?.getToken?.()) {
      return Object.assign({ 'Content-Type': 'application/json' }, window.Auth.getAuthHeaders());
    }
    const t = localStorage.getItem('agroai_jwt') || '';
    return t
      ? { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + t }
      : { 'Content-Type': 'application/json' };
  }

  function _fmt(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  }

  function _fmtShort(iso) {
    if (!iso) return '?';
    return new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
  }

  function _diseaseLabel(d) {
    if (!d) return 'No scan yet';
    return d.split('___').pop().replace(/_/g, ' ');
  }

  function _trendClass(t)  { return 'trend-' + (t || 'stable'); }
  function _trendIcon(t)   { return t === 'recovering' ? '↗' : t === 'worsening' ? '↘' : '→'; }
  function _trendLabel(t)  { return t === 'recovering' ? 'Recovering' : t === 'worsening' ? 'Worsening' : 'Stable'; }

  function _riskPillClass(r) {
    if (r >= 70) return 'risk-high';
    if (r >= 40) return 'risk-med';
    return 'risk-low';
  }

  function _destroyChart(key) {
    if (_charts[key]) { _charts[key].destroy(); delete _charts[key]; }
  }

  // Chart defaults (matches existing Admin/Analytics style)
  function _def() {
    return {
      plugins: { legend: { display: false }, tooltip: { callbacks: {} } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
      },
    };
  }

  // ── API ───────────────────────────────────────────────────────────────────
  async function _fetchPlants() {
    const res  = await fetch('/api/plants', { headers: _authHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to load plants');
    return data;
  }

  async function _fetchHistory(plantId) {
    const res  = await fetch(`/api/plants/${plantId}/history`, { headers: _authHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to load plant history');
    return data;
  }

  async function _apiCreateTrack(plantName) {
    const res  = await fetch('/api/plants/track', {
      method: 'POST', headers: _authHeaders(),
      body: JSON.stringify({ plantName }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to create plant');
    return data;
  }

  async function _apiAddScan(plantId, scanPayload) {
    const res  = await fetch(`/api/plants/${plantId}/scan`, {
      method: 'POST', headers: _authHeaders(),
      body: JSON.stringify(scanPayload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to log scan');
    return data;
  }

  // ── KPI render ────────────────────────────────────────────────────────────
  function _renderKPIs(analytics) {
    const a = analytics || {};
    const setEl = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    setEl('pkpi-total',        a.totalTracked    ?? 0);
    setEl('pkpi-recovery',     a.avgRecoveryRate != null ? a.avgRecoveryRate + '%' : '—');
    setEl('pkpi-high-risk',    a.highRiskPlants  ?? 0);
    setEl('pkpi-most-improved', a.mostImprovedPlant ?? '—');
  }

  // ── Plant Cards ───────────────────────────────────────────────────────────
  function _renderCards(plants) {
    const grid  = document.getElementById('plants-grid');
    const empty = document.getElementById('plants-empty');
    const count = document.getElementById('plants-count');
    if (!grid) return;

    if (count) count.textContent = plants.length
      ? `${plants.length} plant${plants.length !== 1 ? 's' : ''} tracked`
      : '0 plants tracked';

    if (!plants.length) {
      grid.innerHTML = '';
      empty?.classList.remove('hidden');
      return;
    }
    empty?.classList.add('hidden');

    grid.innerHTML = plants.map(p => {
      const trend    = p.trend || 'stable';
      const disease  = _diseaseLabel(p.latestDisease);
      const lastScan = _fmt(p.latestScanDate);
      const risk     = p.latestRiskScore ?? 0;
      const riskCls  = risk >= 70 ? 'sev-high' : risk >= 40 ? 'sev-medium' : 'sev-low';
      return `
        <div class="plant-card" data-plant-id="${p.id}">
          <div class="plant-card-header">
            <div class="plant-card-avatar">🌿</div>
            <div class="plant-card-meta">
              <div class="plant-card-name" title="${p.plantName}">${p.plantName}</div>
              <div class="plant-card-disease">${disease}</div>
            </div>
          </div>
          <div class="plant-card-body">
            <div class="plant-card-stats">
              <div class="plant-card-stat">
                <div class="plant-card-stat-value">${p.totalScans}</div>
                <div class="plant-card-stat-label">Scans</div>
              </div>
              <div class="plant-card-stat">
                <div class="plant-card-stat-value ${riskCls}" style="font-size:.85rem;padding:2px 0">${risk}%</div>
                <div class="plant-card-stat-label">Risk</div>
              </div>
              <div class="plant-card-stat">
                <div class="plant-card-stat-value" style="font-size:.7rem">${lastScan}</div>
                <div class="plant-card-stat-label">Last Scan</div>
              </div>
            </div>
          </div>
          <div class="plant-card-footer">
            <span class="plant-card-time">Since ${_fmt(p.createdAt)}</span>
            <span class="plant-trend-badge ${_trendClass(trend)}">${_trendIcon(trend)} ${_trendLabel(trend)}</span>
          </div>
        </div>`;
    }).join('');

    // Click → open detail
    grid.querySelectorAll('.plant-card').forEach(card => {
      card.addEventListener('click', () => {
        const pid = card.dataset.plantId;
        _openDetail(pid);
        grid.querySelectorAll('.plant-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
      });
    });
  }

  // ── Detail Panel ─────────────────────────────────────────────────────────
  async function _openDetail(plantId) {
    _selectedId = plantId;
    const panel = document.getElementById('plant-detail-panel');
    if (!panel) return;
    panel.classList.remove('hidden');
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Clear old charts
    ['confidence','risk','recovery','health'].forEach(k => _destroyChart(k));

    // Loading state
    document.getElementById('detail-plant-name').textContent = 'Loading…';

    try {
      const data = await _fetchHistory(plantId);
      _populateDetail(data);
    } catch (e) {
      window.Toast?.show('Failed to load plant history: ' + e.message, 'error');
    }
  }

  function _populateDetail(data) {
    const { plant, scans, analytics } = data;
    if (!plant) return;

    // Header
    document.getElementById('detail-plant-name').textContent  = plant.plantName;
    document.getElementById('detail-total-scans').textContent = `${analytics.totalScans} scan${analytics.totalScans !== 1 ? 's' : ''}`;
    document.getElementById('detail-latest-disease').textContent = _diseaseLabel(scans[scans.length - 1]?.disease) || '—';
    document.getElementById('detail-last-scan').textContent   = scans.length ? _fmt(scans[scans.length - 1].scanDate) : '—';

    // Analytics strip
    document.getElementById('det-avg-conf').textContent    = analytics.avgConfidence + '%';
    document.getElementById('det-avg-risk').textContent    = analytics.avgRiskScore + '%';
    document.getElementById('det-recovery').textContent    = analytics.recoveryRate + '%';
    document.getElementById('det-high-risk').textContent   = analytics.highRiskCount;

    // Charts + timeline
    _renderCharts(scans);
    _renderTimeline(scans);
  }

  // ── 4 Chart.js Charts ────────────────────────────────────────────────────
  function _renderCharts(scans) {
    if (!scans?.length) return;
    const labels     = scans.map(s => _fmtShort(s.scanDate));
    const conf       = scans.map(s => s.confidence);
    const risk       = scans.map(s => s.riskScore);
    const recovery   = scans.map(s => Math.max(0, 100 - s.riskScore));
    const health     = scans.map(s => s.healthScore);
    const def        = _def();

    const lineOpts = (bColor, bgColor) => ({
      plugins: def.plugins, scales: def.scales,
      responsive: true, maintainAspectRatio: false,
      elements: { line: { tension: 0.4 } },
    });

    // 1. Confidence
    _destroyChart('confidence');
    const c1 = document.getElementById('chart-confidence-trend');
    if (c1) {
      _charts.confidence = new Chart(c1, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Confidence %', data: conf,
          borderColor: 'rgba(124,58,237,1)', backgroundColor: 'rgba(124,58,237,0.12)',
          borderWidth: 2, fill: true, pointBackgroundColor: 'rgba(124,58,237,1)', pointRadius: 4 }] },
        options: lineOpts(),
      });
    }

    // 2. Risk Trend
    _destroyChart('risk');
    const c2 = document.getElementById('chart-risk-trend');
    if (c2) {
      _charts.risk = new Chart(c2, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Risk Score %', data: risk,
          borderColor: 'rgba(239,68,68,1)', backgroundColor: 'rgba(239,68,68,0.1)',
          borderWidth: 2, fill: true, pointBackgroundColor: 'rgba(239,68,68,1)', pointRadius: 4 }] },
        options: lineOpts(),
      });
    }

    // 3. Recovery Trend (100 - riskScore)
    _destroyChart('recovery');
    const c3 = document.getElementById('chart-recovery-trend');
    if (c3) {
      _charts.recovery = new Chart(c3, {
        type: 'line',
        data: { labels, datasets: [{ label: 'Recovery %', data: recovery,
          borderColor: 'rgba(16,185,129,1)', backgroundColor: 'rgba(16,185,129,0.12)',
          borderWidth: 2, fill: true, pointBackgroundColor: 'rgba(16,185,129,1)', pointRadius: 4 }] },
        options: lineOpts(),
      });
    }

    // 4. Health Score (colored bars)
    _destroyChart('health');
    const c4 = document.getElementById('chart-health-trend');
    if (c4) {
      const bgColors = health.map(h =>
        h >= 70 ? 'rgba(16,185,129,0.75)' : h >= 40 ? 'rgba(245,158,11,0.75)' : 'rgba(239,68,68,0.75)'
      );
      const borderColors = health.map(h =>
        h >= 70 ? 'rgba(16,185,129,1)' : h >= 40 ? 'rgba(245,158,11,1)' : 'rgba(239,68,68,1)'
      );
      _charts.health = new Chart(c4, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Health Score', data: health,
          backgroundColor: bgColors, borderColor: borderColors, borderWidth: 1, borderRadius: 6 }] },
        options: { plugins: def.plugins, scales: def.scales, responsive: true, maintainAspectRatio: false },
      });
    }
  }

  // ── Scan Timeline ─────────────────────────────────────────────────────────
  function _renderTimeline(scans) {
    const tl = document.getElementById('scan-timeline');
    if (!tl) return;
    if (!scans?.length) {
      tl.innerHTML = '<div style="color:var(--text2);font-size:.88rem;text-align:center;padding:24px 0">No scans recorded yet.</div>';
      return;
    }
    // Show newest first in timeline
    const reversed = [...scans].reverse();
    tl.innerHTML = reversed.map((s, i) => {
      const isLast   = i === reversed.length - 1;
      const dotCls   = s.riskScore >= 70 ? 'high-risk' : (s.disease?.toLowerCase().includes('healthy') ? 'healthy' : 'diseased');
      const riskCls  = _riskPillClass(s.riskScore);
      const imgHtml  = s.imageUrl
        ? `<img class="scan-tl-img" src="${s.imageUrl}" alt="scan" onerror="this.style.display='none'">`
        : '';
      return `
        <div class="scan-timeline-item">
          <div class="scan-tl-dot-wrap">
            <div class="scan-tl-dot ${dotCls}"></div>
            ${!isLast ? '<div class="scan-tl-line"></div>' : ''}
          </div>
          <div class="scan-tl-content">
            <div class="scan-tl-disease">${_diseaseLabel(s.disease)}</div>
            <div class="scan-tl-meta">
              <span class="scan-tl-date">${_fmt(s.scanDate)}</span>
              <div class="scan-tl-pills">
                <span class="scan-tl-pill conf">Conf ${s.confidence}%</span>
                <span class="scan-tl-pill ${riskCls}">Risk ${s.riskScore}%</span>
              </div>
            </div>
          </div>
          ${imgHtml}
        </div>`;
    }).join('');
  }

  // ── Track Modal ───────────────────────────────────────────────────────────
  function _openModal(lastScan) {
    _lastScanData = lastScan;
    const modal = document.getElementById('track-modal');
    if (!modal) return;

    // Populate dropdown
    const sel = document.getElementById('track-plant-select');
    if (sel) {
      sel.innerHTML = '<option value="">— Select a plant —</option>' +
        _plants.map(p => `<option value="${p.id}">${p.plantName}</option>`).join('');
    }

    // Pre-fill name from disease
    const nameInput = document.getElementById('track-new-plant-name');
    if (nameInput && lastScan?.disease) {
      const crop = lastScan.disease.split('___')[0].replace(/_/g, ' ');
      nameInput.value = crop ? `My ${_capitalise(crop)} Plant` : '';
    }

    // Hide existing section if no plants yet
    const existSec = document.getElementById('track-existing-section');
    if (existSec) existSec.classList.toggle('hidden', _plants.length === 0);

    document.getElementById('track-error')?.classList.add('hidden');
    modal.classList.add('active');
  }

  function _closeModal() {
    document.getElementById('track-modal')?.classList.remove('active');
  }

  function _capitalise(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function _showModalError(msg) {
    const el = document.getElementById('track-error');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden');
  }

  // ── Public: render (called by Router) ────────────────────────────────────
  async function render() {
    console.log('[PlantTracker] render() called | isLoggedIn:', window.Auth?.isLoggedIn());

    const pagEl   = document.getElementById('page-plants');
    const gate    = document.getElementById('plants-login-gate');
    const content = document.getElementById('plants-content');

    // Belt + suspenders: ensure page itself is not hidden
    if (pagEl) { pagEl.classList.remove('hidden'); pagEl.style.display = ''; }

    if (!window.Auth?.isLoggedIn()) {
      console.log('[PlantTracker] Not logged in — showing login gate');
      if (gate)    { gate.classList.remove('hidden'); gate.style.display = ''; }
      if (content) { content.classList.add('hidden'); content.style.display = 'none'; }
      return;
    }

    // Show content, hide gate
    if (gate)    { gate.classList.add('hidden'); gate.style.display = 'none'; }
    if (content) { content.classList.remove('hidden'); content.style.display = ''; }
    console.log('[PlantTracker] Auth OK — fetching /api/plants');

    try {
      const h = Object.assign({'Content-Type':'application/json'}, window.Auth.getAuthHeaders());
      console.log('[PlantTracker] Authorization header:', h['Authorization']?.slice(0,30) + '...');
      const res  = await fetch('/api/plants', { headers: h });
      console.log('[PlantTracker] /api/plants status:', res.status);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      console.log('[PlantTracker] Response | plants:', data.plants?.length, '| analytics:', JSON.stringify(data.analytics));
      _plants = data.plants || [];
      _renderKPIs(data.analytics || {});
      _renderCards(_plants);
      console.log('[PlantTracker] Render complete');
    } catch (e) {
      console.error('[PlantTracker] Error:', e.message);
      window.Toast?.show('Could not load plants: ' + e.message, 'error');
      _renderCards([]);
    }
  }

  // ── Public: reset (on logout) ─────────────────────────────────────────────
  function reset() {
    _plants      = [];
    _selectedId  = null;
    _lastScanData = null;
    ['confidence','risk','recovery','health'].forEach(k => _destroyChart(k));
    const panel = document.getElementById('plant-detail-panel');
    panel?.classList.add('hidden');
    document.getElementById('plants-grid') && (document.getElementById('plants-grid').innerHTML = '');
  }

  // ── Wire up DOM events ────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {

    // Close detail panel
    document.getElementById('plant-detail-close')?.addEventListener('click', () => {
      document.getElementById('plant-detail-panel')?.classList.add('hidden');
      document.querySelectorAll('.plant-card').forEach(c => c.classList.remove('selected'));
      _selectedId = null;
    });

    // Explicit Plants nav click — safety net in addition to generic [data-page] wiring
    const navPlantsLink = document.getElementById('nav-plants-link');
    if (navPlantsLink) {
      navPlantsLink.addEventListener('click', e => {
        e.preventDefault();
        if (window.Router) Router.navigate('plants');
      });
    }

    // Add New Plant button → open modal with no scan data
    document.getElementById('btn-add-plant')?.addEventListener('click', () => {
      _openModal(null);
    });

    // Track This Plant button (result card) → open modal with scan data
    document.getElementById('track-plant-btn')?.addEventListener('click', () => {
      if (!window.Auth?.isLoggedIn()) {
        window.Toast?.show('Please login to track plants', 'error');
        return;
      }
      // Gather last prediction state from DOM
      const diseaseEl = document.getElementById('prediction-output');
      const riskEl    = document.querySelector('.risk-score-label');
      const confEl    = document.querySelector('.confidence-value');
      const imgEl     = document.querySelector('.image-preview img');
      const disease   = diseaseEl?.textContent?.trim() || 'Unknown';
      const confidence = parseFloat(confEl?.textContent) || 0;
      const riskScore  = parseInt(riskEl?.textContent) || 0;
      const imageUrl   = imgEl?.src || '';

      // Get weather snapshot from DOM
      const weatherSnap = {};
      ['temp','humidity','rain'].forEach(k => {
        const el = document.querySelector(`[data-weather="${k}"]`);
        if (el) weatherSnap[k] = el.textContent;
      });

      _openModal({ disease, confidence, riskScore, imageUrl, weatherSnapshot: weatherSnap });
    });

    // Modal: close button
    document.getElementById('track-modal-close')?.addEventListener('click', _closeModal);
    document.getElementById('track-modal')?.addEventListener('click', e => {
      if (e.target === document.getElementById('track-modal')) _closeModal();
    });

    // Modal: log scan to existing plant
    document.getElementById('track-existing-btn')?.addEventListener('click', async () => {
      const plantId = document.getElementById('track-plant-select')?.value;
      if (!plantId) { _showModalError('Please select a plant.'); return; }
      const btn = document.getElementById('track-existing-btn');
      btn.disabled = true; btn.textContent = 'Saving…';
      try {
        await _apiAddScan(plantId, _lastScanData || {});
        _closeModal();
        window.Toast?.show('✅ Scan logged to plant successfully!', 'success');
        if (Router?.current === 'plants') render();
      } catch (e) {
        _showModalError(e.message);
      } finally {
        btn.disabled = false; btn.textContent = '✅ Log Scan to Selected Plant';
      }
    });

    // Modal: create new plant + log scan
    document.getElementById('track-create-btn')?.addEventListener('click', async () => {
      const name = document.getElementById('track-new-plant-name')?.value?.trim();
      if (!name) { _showModalError('Please enter a plant name.'); return; }
      const btn = document.getElementById('track-create-btn');
      btn.disabled = true; btn.textContent = 'Creating…';
      try {
        const created = await _apiCreateTrack(name);
        if (_lastScanData) {
          await _apiAddScan(created.plantId, _lastScanData);
        }
        _closeModal();
        window.Toast?.show(`🌱 "${name}" created and scan logged!`, 'success');
        if (Router?.current === 'plants') render();
        else { _plants.push({ id: created.plantId, plantName: name, totalScans: _lastScanData ? 1 : 0, trend: 'stable', latestRiskScore: 0, latestDisease: null, latestScanDate: null, createdAt: new Date().toISOString() }); }
      } catch (e) {
        _showModalError(e.message);
      } finally {
        btn.disabled = false; btn.textContent = '🌱 Create & Track';
      }
    });

    // Go scan button in empty state
    document.getElementById('plants-go-scan-btn')?.addEventListener('click', () => Router?.navigate('home'));

    // Login gate button — use Auth.showLogin() directly
    document.getElementById('plants-gate-login-btn')?.addEventListener('click', () => {
      window.Auth?.showLogin?.();
    });
  });

  return { render, reset };
})();

// Expose on window so auth handlers and future code can reference it safely
window.PlantTracker = PlantTracker;

// Plants page navigation is handled in the _patchRouter() IIFE above.
// Router.pages already includes 'plants' and PlantTracker.render() is called there.
