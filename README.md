# AgroAI 🌾

> **AI-Powered Agricultural Intelligence Platform**

[![Version](https://img.shields.io/badge/version-2.1.0-brightgreen)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-3.x-black)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/mongodb-atlas-green)](https://cloud.mongodb.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

AgroAI is a production-grade, full-stack agricultural intelligence platform designed to empower farmers, agronomists, and researchers with real-time disease detection, weather-informed risk analysis, longitudinal plant monitoring, and an AI-powered expert assistant grounded in verified agricultural knowledge.

---

## ✨ Features

| Category | Feature | Status |
|---|---|---|
| **AI Detection** | Plant Disease Detection (38 classes) | ✅ |
| **AI Detection** | Disease Severity Analysis (LOW / MEDIUM / HIGH) | ✅ |
| **Authentication** | JWT Authentication & Session Management | ✅ |
| **SaaS** | Multi-User SaaS Architecture | ✅ |
| **Dashboard** | Personalised User Dashboard | ✅ |
| **Analytics** | Analytics Dashboard (KPIs, Charts, Trends) | ✅ |
| **Weather** | Weather Intelligence (OpenWeatherMap) | ✅ |
| **Weather** | Geolocation Weather (GPS Coordinates) | ✅ |
| **Insights** | Smart Farming Insights from Weather | ✅ |
| **Risk** | Disease Spread Risk Prediction Engine | ✅ |
| **Language** | Multi-Language Support (English / हिंदी) | ✅ |
| **Feedback** | MongoDB-backed Feedback System | ✅ |
| **RAG** | RAG Agriculture Expert Assistant | ✅ |
| **RAG** | FAISS Vector Search (Semantic Retrieval) | ✅ |
| **RAG** | Agriculture Knowledge Base (7 domains) | ✅ |
| **RAG** | Source Citations from Knowledge Base | ✅ |
| **Reports** | Premium PDF Report Generation | ✅ |
| **Admin** | Admin Intelligence Dashboard | ✅ |
| **Tracking** | 🆕 Plant Disease Progress Tracking | ✅ |
| **Tracking** | 🆕 Longitudinal Scan History (per plant) | ✅ |
| **Tracking** | 🆕 Recovery & Risk Trend Visualization | ✅ |
| **Tracking** | 🆕 Plant Monitoring Analytics Dashboard | ✅ |

---

## 🌿 Phase 4B — Disease Progress Tracking

> **New in v2.1.0** — Track the same plant over time and visualize its disease recovery or progression.

### Overview

Instead of treating every scan as isolated, AgroAI now lets users **create tracked plants**, log multiple scans over time, and visualize:

- **Disease Confidence** — how confident the AI is across scans
- **Risk Score History** — is the plant getting better or worse?
- **Recovery Trend** — computed as `100 - riskScore` over time
- **Health Score** — color-coded composite health indicator

### Features

- **🌿 Track This Plant** — Button appears after every AI prediction to immediately log the scan to a tracked plant
- **Plant Cards** — Visual dashboard of all tracked plants with latest disease, scan count, and trend badge (Recovering / Worsening / Stable)
- **Recovery Dashboard** — 4 Chart.js charts per plant (Confidence, Risk, Recovery, Health Score)
- **Scan Timeline** — Chronological history of all scans with disease label, confidence, risk, and image
- **Add New Plant** — Create a tracked plant directly from the Plants page
- **Plant Analytics KPIs** — Total tracked, average recovery rate, high-risk plants, most improved plant

### Architecture

```
Plant Image Scan (AI Prediction)
          ↓
  "Track This Plant" Modal
          ↓
  Select Existing Plant  OR  Create New Tracked Plant
          ↓
  POST /api/plants/track        POST /api/plants/<id>/scan
          ↓                              ↓
  plant_tracks collection     tracked_plant_scans collection
          ↓
  MongoDB Atlas (user-scoped)
          ↓
  GET /api/plants/<id>/history
          ↓
  Recovery Analytics + 4 Trend Charts
          ↓
  Chart.js Visualization (Confidence / Risk / Recovery / Health Score)
```

### MongoDB Collections

| Collection | Purpose |
|---|---|
| `plant_tracks` | One document per tracked plant (name, userId, createdAt, summary) |
| `tracked_plant_scans` | One document per scan snapshot (disease, confidence, riskScore, weather, imageUrl, scanDate) |

### API Endpoints (Phase 4B)

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/plants/track` | Create a new tracked plant | JWT Required |
| `POST` | `/api/plants/<id>/scan` | Log a scan to a tracked plant | JWT Required |
| `GET` | `/api/plants` | List all plants + analytics | JWT Required |
| `GET` | `/api/plants/<id>/history` | Full history + trend charts data | JWT Required |

### Screenshots

| View | Preview |
|---|---|
| 🌿 **Plant Monitoring Dashboard** | *(screenshot placeholder)* |
| 🔁 **Track This Plant Modal** | *(screenshot placeholder)* |
| 📈 **Recovery Analytics (4 Charts)** | *(screenshot placeholder)* |
| 🕐 **Scan Timeline** | *(screenshot placeholder)* |

---

## 🏗️ Architecture

### Plant Disease Detection Pipeline

```
Plant Image (Upload / URL)
        ↓
TensorFlow Model (128×128 CNN)
        ↓
Disease Detection  →  38 Plant Disease Classes
        ↓
Weather Analysis  →  OpenWeatherMap API
        ↓
Risk Prediction   →  Rule-Based Risk Engine
        ↓
MongoDB Storage   →  Atlas Cloud Database
        ↓
Dashboard Analytics  →  User KPIs & Trends
        ↓
Plant Monitoring  →  Progress Tracking & Recovery Trends
```

### RAG Agriculture Expert Pipeline

```
User Question
      ↓
FAISS Vector Search  →  all-MiniLM-L6-v2 Embeddings
      ↓
Knowledge Base  →  ICAR / Agriculture Domain Docs
      ↓
Grounded Prompt  →  Retrieved Context (Top-5 Chunks)
      ↓
Gemini 1.5 Flash
      ↓
Grounded Answer + Source Citations
```

---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| HTML5 | Structure & Semantic Markup |
| CSS3 (Vanilla) | Styling, Animations, Glassmorphism |
| JavaScript (ES6+) | Dynamic UI, Fetch API, i18n Engine |
| Chart.js | Analytics Charts & Trend Visualization |

### Backend
| Technology | Purpose |
|---|---|
| Flask 3.x | Web Framework & REST API |
| MongoDB Atlas | Cloud Database (Scans, Users, Feedback, Plant Tracks) |
| Flask-JWT-Extended | Multi-User Authentication & Authorization |
| TensorFlow 2.x + Keras | Plant Disease Classification Model |
| Google Gemini 1.5 Flash | AI Chatbot & RAG Answer Generation |
| Gunicorn | Production WSGI Server |
| ReportLab | PDF Report Generation |

### AI / ML
| Technology | Purpose |
|---|---|
| FAISS | High-Performance Vector Similarity Search |
| Sentence Transformers | Semantic Embedding (`all-MiniLM-L6-v2`) |
| LangChain | RAG Pipeline Orchestration |
| PyPDF | PDF Document Ingestion for Knowledge Base |

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (free tier works)
- Google Gemini API key — [get it here](https://aistudio.google.com/app/apikey)
- OpenWeatherMap API key — [get it here](https://openweathermap.org/api)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/agroai.git
cd agroai
```

### 2. Create a Virtual Environment & Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your real credentials (see [Environment Variables](#-environment-variables)).

### 4. Build the Agriculture Knowledge Base Index

```bash
python ingest.py
```

> ⏱️ First run takes ~2–3 minutes to download the embedding model.

### 5. Start the Application

```bash
python app.py
```

The server will start at **http://127.0.0.1:5000**

> For production, use Gunicorn: `gunicorn -w 2 -b 0.0.0.0:5000 app:app`

---

## 🔑 Environment Variables

```env
# ── Google Gemini AI ─────────────────────────────────────────
GOOGLE_API_KEY=your_google_gemini_api_key_here

# ── MongoDB Atlas ─────────────────────────────────────────────
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/agroai?retryWrites=true&w=majority

# ── JWT Authentication ────────────────────────────────────────
JWT_SECRET_KEY=change_this_to_a_strong_random_secret_min_32_chars
JWT_ACCESS_TOKEN_EXPIRES=86400

# ── OpenWeatherMap (optional) ─────────────────────────────────
OPENWEATHER_API_KEY=your_openweathermap_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already git-ignored.

---

## 📁 Project Structure

```
agroai/
├── app.py                          # Main Flask application & all API routes
├── database.py                     # MongoDB operations (scans, users, plant tracks)
├── risk_engine.py                  # Disease spread risk calculation engine
├── pdf_generator.py                # Premium PDF report generation (ReportLab)
├── ingest.py                       # RAG knowledge base ingestion & FAISS indexing
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
│
├── auth/                           # JWT Authentication module
│   ├── auth_routes.py              #   /api/auth/register, /login, /me, /profile
│   └── auth_utils.py              #   Password hashing, JWT helpers
│
├── services/                       # Business logic services
│   ├── weather_service.py          #   OpenWeatherMap API integration
│   ├── farming_insights.py         #   AI-generated farming tips from weather
│   └── rag_service.py              #   FAISS + Gemini RAG pipeline
│
├── knowledge_base/                 # Agriculture domain documents (7 categories)
│   ├── crops/                      #   Crop cultivation guides
│   ├── diseases/                   #   Plant disease reference docs
│   ├── fertilizers/                #   Fertilizer usage & schedules
│   ├── government/                 #   Government schemes & subsidies
│   ├── irrigation/                 #   Irrigation methods & water management
│   ├── pesticides/                 #   Pesticide safety & application
│   └── weather/                    #   Weather-based farming advice
│
├── vectorstore/                    # Auto-generated FAISS index (run ingest.py)
│   ├── index.faiss                 #   FAISS binary index
│   └── index.pkl                   #   LangChain metadata & docstore
│
├── translations/                   # Multi-language support
│   ├── en.json                     #   English UI strings
│   └── hi.json                     #   Hindi (हिंदी) UI strings
│
├── index.html                      # Single-page frontend (SPA)
├── script.js                       # Frontend JS (auth, scans, RAG, i18n, PlantTracker)
├── style.css                       # Styling (glassmorphism, animations, dark UI)
├── auth.js                         # Frontend authentication module
│
├── disease_info.json               # Local disease knowledge base (38 diseases)
├── disease_rules.json              # Risk engine rules
├── trained_plant_disease_model.keras  # TensorFlow CNN model
└── settings.json                   # App settings
```

---

## 🔌 API Reference

### Authentication
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/auth/register` | Register new user | Public |
| `POST` | `/api/auth/login` | Login & get JWT | Public |
| `GET` | `/api/auth/me` | Get current user profile | JWT |

### Disease Detection
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/predict` | Analyze plant image for disease | Optional JWT |
| `GET` | `/api/scans` | Get scan history | Optional JWT |
| `DELETE` | `/api/scans/<id>` | Delete a scan | Optional JWT |
| `DELETE` | `/api/scans` | Clear all scans | Optional JWT |

### Dashboard & Analytics
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/api/dashboard` | Personalized KPIs & trends | JWT Required |

### Weather & Risk
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/api/weather?city=<city>` | Get weather by city name | Public |
| `GET` | `/api/weather?lat=<lat>&lon=<lon>` | Get weather by GPS coordinates | Public |
| `POST` | `/api/risk-analysis` | Disease spread risk analysis | Public |

### Feedback
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/feedback` | Submit feedback (rating + message) | JWT Required |
| `GET` | `/api/feedback/my` | Get my feedback submissions | JWT Required |
| `GET` | `/api/feedback/stats` | Aggregate feedback statistics | Public |

### RAG Expert Assistant
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/rag-chat` | Ask agriculture expert question | Optional JWT |
| `GET` | `/api/rag-chat/history` | Get RAG chat history | Optional JWT |
| `POST` | `/api/admin/rebuild-index` | Hot-rebuild FAISS index | JWT Required |

### 🌿 Plant Progress Tracking (Phase 4B)
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/plants/track` | Create a new tracked plant | JWT Required |
| `POST` | `/api/plants/<id>/scan` | Log a scan snapshot to a plant | JWT Required |
| `GET` | `/api/plants` | List all plants + analytics KPIs | JWT Required |
| `GET` | `/api/plants/<id>/history` | Full history + chart data for a plant | JWT Required |

### Reports & Chat
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/generate-report` | Generate premium PDF report | Optional JWT |
| `POST` | `/chat` | Disease-context Gemini chatbot | Public |

---

## 📸 Screenshots

> Screenshots will be added after the live deployment.

| View | Preview |
|---|---|
| 🏠 **Home / Landing** | *(screenshot placeholder)* |
| 🔬 **Disease Detection** | *(screenshot placeholder)* |
| ⛅ **Weather Dashboard** | *(screenshot placeholder)* |
| 📊 **Analytics Dashboard** | *(screenshot placeholder)* |
| 💬 **Feedback Panel** | *(screenshot placeholder)* |
| 🤖 **RAG Expert Assistant** | *(screenshot placeholder)* |
| 🛡️ **Admin Intelligence Dashboard** | *(screenshot placeholder)* |
| 🌿 **Plant Monitoring Dashboard** | *(screenshot placeholder)* |
| 🔁 **Track This Plant Modal** | *(screenshot placeholder)* |
| 📈 **Recovery Analytics** | *(screenshot placeholder)* |

---

## 🔒 Security

- **`.env` is git-ignored** — credentials never enter version control
- **JWT tokens** expire after 24 hours (configurable via `JWT_ACCESS_TOKEN_EXPIRES`)
- **Password hashing** via `bcrypt` — plain-text passwords never stored
- **User isolation** — all MongoDB queries scoped to authenticated `user_id`
- **Plant data isolation** — `plant_tracks` and `tracked_plant_scans` are strictly user-scoped
- **Input validation** on all endpoints

---

## 🌐 Multi-Language Support

AgroAI supports **English** and **Hindi (हिंदी)** throughout the entire interface:

- UI strings loaded dynamically from `translations/en.json` and `translations/hi.json`
- Chatbot and RAG assistant respond in the selected language
- Language toggle persists across sessions

---

## 📊 Analytics Dashboard

The personalized dashboard provides:

- **Total Scans** — cumulative disease detection count
- **Healthy vs. Diseased** — split with percentage
- **Average Confidence** — model accuracy across all scans
- **Highest Risk Scan** — most critical detection on record
- **30-Day Activity Chart** — monthly scan frequency trend
- **Disease Distribution** — top detected diseases breakdown

---

## 🌿 Plant Monitoring Dashboard

The Plant Monitoring page provides cross-plant analytics:

- **Total Tracked Plants** — number of plants under longitudinal monitoring
- **Average Recovery Rate** — mean `100 - avgRiskScore` across all plants
- **High Risk Plants** — count of plants with latest risk score ≥ 70
- **Most Improved Plant** — plant with the largest risk score drop over time

Per-plant recovery dashboard includes:

- 📊 **Disease Confidence Trend** — Line chart of AI confidence over scans
- ⚠️ **Risk Score Trend** — Line chart of disease risk over time
- 📈 **Recovery Trend** — Line chart of `100 - riskScore` (higher = healthier)
- 💚 **Health Score Trend** — Color-coded bar chart (green/amber/red by health)

---

## 🤖 RAG Knowledge Base

The agriculture knowledge base spans **7 expert domains**:

| Domain | Coverage |
|---|---|
| 🌾 Crops | Cultivation guides, sowing seasons, varieties |
| 🦠 Diseases | Disease profiles, pathogens, symptoms |
| 🧪 Fertilizers | NPK recommendations, application timing |
| 🏛️ Government Schemes | PM-KISAN, crop insurance, subsidies |
| 💧 Irrigation | Drip, sprinkler, flood irrigation best practices |
| 🐛 Pesticides | Safe usage, dosages, integrated pest management |
| ⛅ Weather | Seasonal farming advice, climate adaptation |

---

## 🚢 Release History

### v2.1.0 — Plant Monitoring & Disease Progress Tracking *(Current)*
- ✅ Phase 4B: Longitudinal Plant Disease Progress Tracking
- ✅ `plant_tracks` & `tracked_plant_scans` MongoDB collections
- ✅ 4 JWT-protected plant tracking API endpoints
- ✅ "Track This Plant" button in AI prediction result
- ✅ Track Plant modal (log to existing or create new)
- ✅ Plant Monitoring page with KPI cards
- ✅ Recovery dashboard with 4 Chart.js trend charts
- ✅ Scan timeline with disease labels, confidence, and risk pills
- ✅ Cross-plant analytics (avg recovery rate, high-risk count, most improved)

### v2.0.0 — RAG Agriculture Expert & Admin Dashboard
- ✅ Phase 4A: Admin Intelligence Dashboard (role-based)
- ✅ Phase 3A: RAG Agriculture Expert Assistant
- ✅ FAISS Vector Search with semantic retrieval
- ✅ 7-domain Agriculture Knowledge Base
- ✅ Chat History persisted to MongoDB
- ✅ Source Citations from knowledge documents
- ✅ Phase 2A: Multi-Language Support (EN / हिंदी)
- ✅ Phase 2B: MongoDB Feedback System
- ✅ Phase 2C: Enhanced Analytics Dashboard

### v1.0.0 — Multi-User SaaS Platform
- ✅ JWT Authentication & Multi-User SaaS
- ✅ Weather Intelligence & Geolocation Weather
- ✅ Disease Spread Risk Prediction Engine
- ✅ Analytics Dashboard & Personalized Dashboard
- ✅ Premium PDF Report Generation
- ✅ MongoDB Atlas Integration

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for Indian farmers**

[Report Bug](https://github.com) · [Request Feature](https://github.com) · [Documentation](https://github.com)

</div>
