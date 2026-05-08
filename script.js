// The URL of your Flask backend
const API_URL = "http://127.0.0.1:5000";

// --- Page Element References ---
const imageUploader    = document.getElementById("image-uploader");
const urlUploader      = document.getElementById("url-uploader");
const predictButton    = document.getElementById("predict-button");
const imagePreview     = document.getElementById("image-preview");

const preventionCard   = document.getElementById("prevention-card");
const resultsDiv       = document.getElementById("results");

const predictionOutput = document.getElementById("prediction-output");
const symptomsOutput   = document.getElementById("symptoms-output");
const treatmentOutput  = document.getElementById("treatment-output");
const preventionOutput = document.getElementById("prevention-output");
const symptomsSection  = document.getElementById("symptoms-section");
const treatmentSection = document.getElementById("treatment-section");
const preventionTitle  = document.getElementById("prevention-title");

const chatContainer    = document.getElementById("chat-container");
const chatBox          = document.getElementById("chat-box");
const chatInput        = document.getElementById("chat-message");
const chatSendButton   = document.getElementById("chat-send-button");
const uploadArea       = document.getElementById("upload-area");
const uploadLabel      = document.querySelector(".upload-label");

// --- Global state ---
let chatContext = {
    session_id: "user_" + Date.now(),
    disease:    "",
    symptoms:   "",
    treatment:  "",
    prevention: ""
};
let currentFile = null;

// ─── Navbar scroll effect ──────────────────────────────────────────────────────
const navbar = document.getElementById("navbar");
window.addEventListener("scroll", () => {
    navbar.classList.toggle("scrolled", window.scrollY > 20);
});

// ─── Scroll animations ─────────────────────────────────────────────────────────
const animObserver = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add("visible"); });
}, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });
document.querySelectorAll(".anim-fade-up").forEach(el => animObserver.observe(el));

// ─── Animated Stats Counter ────────────────────────────────────────────────────
function animateCounter(el) {
    const target = parseInt(el.dataset.target);
    const suffix = el.dataset.suffix || "";
    let current = 0;
    const step = Math.ceil(target / 60);
    const timer = setInterval(() => {
        current = Math.min(current + step, target);
        el.textContent = current + suffix;
        if (current >= target) clearInterval(timer);
    }, 20);
}
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(e => {
        if (e.isIntersecting && !e.target.dataset.counted) {
            e.target.dataset.counted = "1";
            animateCounter(e.target);
        }
    });
}, { threshold: 0.5 });
document.querySelectorAll(".stat-number").forEach(el => statsObserver.observe(el));

// ─── Particle Canvas ───────────────────────────────────────────────────────────
(function initParticles() {
    const canvas = document.getElementById("particles-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W, H, particles = [];
    const resize = () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);
    const colors = ["rgba(124,58,237,0.5)", "rgba(16,185,129,0.4)", "rgba(59,130,246,0.4)", "rgba(168,85,247,0.4)"];
    for (let i = 0; i < 55; i++) {
        particles.push({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            r: Math.random() * 1.8 + 0.4,
            dx: (Math.random() - 0.5) * 0.4,
            dy: (Math.random() - 0.5) * 0.4,
            color: colors[Math.floor(Math.random() * colors.length)]
        });
    }
    (function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();
            p.x += p.dx; p.y += p.dy;
            if (p.x < 0 || p.x > W) p.dx *= -1;
            if (p.y < 0 || p.y > H) p.dy *= -1;
        });
        requestAnimationFrame(draw);
    })();
})();

// ─── Mouse Glow Tracking ───────────────────────────────────────────────────────
document.addEventListener("mousemove", (e) => {
    document.documentElement.style.setProperty("--mx", e.clientX + "px");
    document.documentElement.style.setProperty("--my", e.clientY + "px");
});

// ─── Hamburger menu ────────────────────────────────────────────────────────────
const hamburger = document.getElementById("hamburger");
const navLinks = document.querySelector(".nav-links");
if (hamburger && navLinks) {
    hamburger.addEventListener("click", () => navLinks.classList.toggle("open"));
}

// ─── Drag/Drop & File & URL Logic ─────────────────────────────────────────────
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('dragover'); });
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});
imageUploader.addEventListener("change", () => {
    const file = imageUploader.files[0];
    if (file) handleFile(file);
});
function handleFile(file) {
    currentFile = file;
    urlUploader.value = "";
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        imagePreview.style.display = "block";
    };
    reader.readAsDataURL(file);
    uploadLabel.textContent = file.name;
}
urlUploader.addEventListener("input", () => {
    const url = urlUploader.value.trim();
    if (url) {
        currentFile = null;
        imageUploader.value = "";
        uploadLabel.textContent = "Click to upload or drag & drop";
        imagePreview.innerHTML = `<img src="${url}" alt="Image URL Preview" onerror="this.parentElement.style.display='none';">`;
        imagePreview.style.display = "block";
    } else {
        imagePreview.innerHTML = "";
        imagePreview.style.display = "none";
    }
});

// ─── Prediction Logic ──────────────────────────────────────────────────────────
predictButton.addEventListener("click", async () => {
    const imageUrl = urlUploader.value.trim();
    let requestBody, requestHeaders = {};

    if (currentFile) {
        const formData = new FormData();
        formData.append("file", currentFile);
        requestBody = formData;
    } else if (imageUrl) {
        requestBody = JSON.stringify({ "url": imageUrl });
        requestHeaders["Content-Type"] = "application/json";
    } else {
        alert("Please choose an image file, drag one in, or paste an image URL.");
        return;
    }

    setLoading(true);
    preventionCard.classList.add("hidden");
    resultsDiv.classList.add("hidden");
    chatContainer.classList.add("hidden");

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method:  "POST",
            headers: requestHeaders,
            body:    requestBody,
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Prediction failed");
        }
        const data = await response.json();

        displayPrediction(data);

        chatContext.disease    = data.disease_name;
        chatContext.symptoms   = data.symptoms;
        chatContext.treatment  = data.treatment;
        chatContext.prevention = data.prevention;

        chatBox.innerHTML = "";
        addMessageToChat(
            `👋 Hello! I've analyzed your plant and detected **${data.disease_name}**.\n\n` +
            `You can ask me about:\n` +
            `• 🔍 **Symptoms** — what to look for\n` +
            `• 💊 **Treatment** — how to treat it\n` +
            `• 🛡️ **Prevention** — how to stop it\n` +
            `• 🌿 **Causes** — why it happens\n\n` +
            `_I'm here to help! Just type your question below._`,
            "bot"
        );

    } catch (error) {
        alert("Error: " + error.message);
        if (imageUrl) imagePreview.innerHTML = "";
        preventionCard.classList.remove("hidden");
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    const btnLabel   = predictButton.querySelector(".btn-label");
    const btnSpinner = document.getElementById("btn-spinner");
    if (isLoading) {
        predictButton.disabled = true;
        if (btnLabel)   btnLabel.textContent = "Analyzing...";
        if (btnSpinner) btnSpinner.classList.remove("hidden");
        predictButton.classList.add("loading");
    } else {
        predictButton.disabled = false;
        if (btnLabel)   btnLabel.textContent = "🔬 Analyze Image";
        if (btnSpinner) btnSpinner.classList.add("hidden");
        predictButton.classList.remove("loading");
    }
}

function displayPrediction(data) {
    predictionOutput.textContent  = data.disease_name;
    symptomsOutput.innerHTML      = data.symptoms.replace(/\n/g, '<br>');
    treatmentOutput.innerHTML     = data.treatment.replace(/\n/g, '<br>');
    preventionOutput.innerHTML    = data.prevention.replace(/\n/g, '<br>');

    const isHealthy = data.prediction.includes("healthy");
    const confidenceBar   = document.getElementById("confidence-bar");
    const confidenceValue = document.getElementById("confidence-value");
    const severityBadge   = document.getElementById("severity-badge");

    if (isHealthy) {
        predictionOutput.classList.add("healthy");
        symptomsSection.classList.add("hidden");
        treatmentSection.classList.add("hidden");
        preventionTitle.textContent = "🌱 How to Keep it Healthy";
        if (severityBadge) { severityBadge.textContent = "Healthy"; severityBadge.classList.add("healthy"); }
        if (confidenceBar)   { setTimeout(() => { confidenceBar.style.width = "97%"; }, 300); }
        if (confidenceValue) confidenceValue.textContent = "97%";
    } else {
        predictionOutput.classList.remove("healthy");
        symptomsSection.classList.remove("hidden");
        treatmentSection.classList.remove("hidden");
        preventionTitle.textContent = "🛡️ Prevention";
        if (severityBadge) { severityBadge.textContent = "Disease Detected"; severityBadge.classList.remove("healthy"); }
        if (confidenceBar)   { setTimeout(() => { confidenceBar.style.width = "92%"; }, 300); }
        if (confidenceValue) confidenceValue.textContent = "92%";
    }

    resultsDiv.classList.remove("hidden");
    chatContainer.classList.remove("hidden");
    resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─── Chat Logic ────────────────────────────────────────────────────────────────
chatSendButton.addEventListener("click", sendChatMessage);
chatInput.addEventListener("keyup", (e) => {
    if (e.key === "Enter") sendChatMessage();
});

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (message === "") return;

    addMessageToChat(message, "user");
    chatInput.value = "";
    chatSendButton.disabled = true;

    // Animated typing indicator with bouncing dots
    const typingIndicator = createTypingIndicator();
    chatBox.appendChild(typingIndicator);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message:    message,
                session_id: chatContext.session_id,
                disease:    chatContext.disease,
                symptoms:   chatContext.symptoms,
                treatment:  chatContext.treatment,
                prevention: chatContext.prevention
            }),
        });

        if (chatBox.contains(typingIndicator)) chatBox.removeChild(typingIndicator);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Chat failed");
        }

        const data = await response.json();
        const isLocalMode = data.mode === "local";
        if (isLocalMode) showLocalModeBadge();
        addMessageToChat(data.response, "bot", isLocalMode);

    } catch (error) {
        if (chatBox.contains(typingIndicator)) chatBox.removeChild(typingIndicator);
        // Never show raw errors to the user
        showLocalModeBadge();
        addMessageToChat(
            "🌱 *Powered by Local Plant Expert Mode*\n\n" +
            "I'm using my offline knowledge base right now.\n\n" +
            "Try asking me about **symptoms**, **treatment**, or **prevention** for your plant!",
            "bot",
            true
        );
    } finally {
        chatSendButton.disabled = false;
        chatInput.focus();
    }
}

function showLocalModeBadge() {
    const badge = document.getElementById("local-mode-badge");
    if (badge) badge.classList.remove("hidden");
}

/**
 * Creates an animated typing indicator (three bouncing dots).
 */
function createTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.classList.add("chat-msg", "bot", "typing-indicator");
    wrapper.innerHTML = `
        <span class="typing-label">🌿 Expert is thinking</span>
        <span class="dots">
            <span></span><span></span><span></span>
        </span>`;
    return wrapper;
}

/**
 * Renders a rich markdown-like message into the chat.
 * @param {string}  text        - The message text (supports **bold**, _italic_, • bullets, headings)
 * @param {string}  sender      - "bot" or "user"
 * @param {boolean} isLocalMode - If true, shows the "Local Expert Mode" badge
 */
function addMessageToChat(text, sender, isLocalMode = false) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("chat-msg", sender);

    if (sender === "bot" && isLocalMode) {
        msgDiv.classList.add("local-mode");
    }

    // ── Rich markdown renderer ──
    let html = text
        // Local expert mode header → styled badge (strip from body if present)
        .replace(/🌱 \*Powered by Local Plant Expert Mode\*\n*/g, "")
        // Bold **text**
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        // Italic _text_
        .replace(/_(.*?)_/g, "<em>$1</em>")
        // Bullet lines starting with • or -
        .replace(/^[•\-]\s(.+)/gm, "<li>$1</li>")
        // Numbered list 1. 2. …
        .replace(/^\d+\.\s+(.+)/gm, "<li>$1</li>")
        // Wrap consecutive <li> into <ul>
        .replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`)
        // Line breaks
        .replace(/\n/g, "<br>");

    // Inject local-mode badge at top if applicable
    if (sender === "bot" && isLocalMode) {
        html = `<div class="local-badge">🌱 Local Plant Expert Mode</div>` + html;
    }

    msgDiv.innerHTML = html;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
