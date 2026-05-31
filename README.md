<div align="center">

# 🌿 AgroAI — Intelligent Plant Disease Detection Platform

**Phase 1 Complete · Multi-User SaaS · AI-Powered · Real-Time Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> AgroAI is a production-grade agricultural intelligence SaaS platform that combines **TensorFlow deep learning**, **Google Gemini AI**, **real-time weather data**, and a **predictive disease risk engine** to help farmers detect plant diseases, understand risk, and take action — all from their browser.

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **AI Disease Detection** | TensorFlow CNN model trained on 87,000+ plant images across 38 disease classes |
| 🔐 **JWT Authentication** | Secure register/login/logout with bcrypt password hashing |
| 👤 **Multi-User SaaS** | Full data isolation — each user sees only their own scans, history, and analytics |
| ☁️ **MongoDB Atlas** | Cloud-persisted scan history with user-scoped queries |
| 🌦️ **Weather Intelligence** | Real-time weather via OpenWeatherMap API with 30-min cache and simulated fallback |
| 📍 **Geolocation Weather** | Auto-detects user's GPS location via `navigator.geolocation` |
| 🦠 **Disease Risk Engine** | Fuzzy-logic risk scoring (HIGH/MEDIUM/LOW) based on disease type + weather conditions |
| 📊 **Analytics Dashboard** | Personalised KPIs, risk distribution charts, and activity timeline per user |
| 💬 **Gemini AI Chatbot** | Context-aware plant disease assistant powered by Google Gemini 1.5 Flash |
| 📄 **PDF Reports** | Premium AI-generated scan reports via ReportLab |
| 🌍 **Farming Insights** | 9 threshold-based agronomic advisory rules from weather data |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AgroAI Platform                         │
├───────────────┬─────────────────────┬───────────────────────┤
│   Frontend    │      Backend        │     Intelligence       │
│               │                    │                        │
│  HTML5/CSS3   │   Flask 3.x        │  TensorFlow CNN        │
│  Vanilla JS   │   Flask-JWT-Extended│  Gemini 1.5 Flash      │
│  Chart.js     │   PyMongo          │  Risk Engine (Fuzzy)   │
│  Glassmorphism│   Flask-Bcrypt     │  Weather Service (OWM) │
│               │   ReportLab        │  Farming Insights      │
└───────────────┴─────────────────────┴───────────────────────┘
                           │
                    MongoDB Atlas
                  (User-Scoped Data)
```

### Data Flow — Authenticated Scan

```
Browser (logged in)
  → POST /predict  +  Authorization: Bearer <JWT>
  → get_optional_user_id()  →  ObjectId("user123")
  → TensorFlow prediction  →  disease_name, confidence
  → get_weather("Bhopal")  →  humidity, temperature
  → risk_engine.analyse_risk()  →  HIGH/MEDIUM/LOW
  → MongoDB: { disease, confidence, userId, riskLevel, weatherSnapshot }
  → Response: { disease_name, riskLevel, riskScore, riskReason }
  → Frontend: History page updates immediately ✅
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- MongoDB Atlas cluster (free tier works)
- Google Gemini API key ([get one](https://aistudio.google.com/app/apikey))
- OpenWeatherMap API key ([get one](https://openweathermap.org/api)) *(optional — has simulation fallback)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/PDDS.git
cd PDDS/plant-disease-detection-system

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your actual keys (see Environment Variables section)

# 5. Run the application
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🔐 Environment Variables

Create a `.env` file in the project root (never commit this file):

```env
# Google Gemini AI
GOOGLE_API_KEY=your_gemini_api_key_here

# MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/agroai?retryWrites=true&w=majority

# JWT Authentication
JWT_SECRET_KEY=your_strong_random_secret_here_min_32_chars
JWT_ACCESS_TOKEN_EXPIRES=86400

# OpenWeatherMap (optional — falls back to simulation if missing)
OPENWEATHER_API_KEY=your_openweathermap_api_key_here
```

> ⚠️ **Security**: The `.env` file is listed in `.gitignore` and will never be committed. All secrets must live here only.

---

## 📁 Project Structure

```
plant-disease-detection-system/
├── app.py                    # Flask application & all API routes
├── database.py               # MongoDB Atlas CRUD operations (user-scoped)
├── risk_engine.py            # Fuzzy-logic disease spread risk scoring
├── pdf_generator.py          # ReportLab PDF report generation
├── disease_rules.json        # 38-class disease→risk threshold mapping
├── auth/
│   └── routes.py             # JWT auth blueprints (register/login/logout/profile)
├── services/
│   ├── weather_service.py    # OpenWeatherMap + GPS coord lookup + fallback
│   └── farming_insights.py  # Agronomic threshold-based advisory rules
├── index.html                # Single-page application shell
├── script.js                 # Frontend modules (Router, Storage, Analytics, Weather, RiskCard)
├── auth.js                   # JWT auth state manager (window.Auth)
├── style.css                 # Glassmorphic CSS design system
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Excludes .env, __pycache__, venv, etc.
└── trained_plant_disease_model.keras  # TensorFlow CNN (38 classes)
```

---

## 🔌 API Reference

### Authentication

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/register` | None | Create new account |
| `POST` | `/api/auth/login` | None | Login, returns JWT |
| `POST` | `/api/auth/logout` | JWT | Invalidate session |
| `GET` | `/api/auth/profile` | JWT | Get user profile |
| `PUT` | `/api/auth/profile` | JWT | Update profile |

### Core AI

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/predict` | Optional | Run disease detection + risk analysis |
| `POST` | `/chat` | None | Gemini AI chatbot |
| `POST` | `/generate-report` | None | Generate PDF report |

### Data & Intelligence

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/scans` | Optional | Get scan history (user-scoped if JWT present) |
| `DELETE` | `/api/scans/<id>` | Optional | Delete a scan |
| `GET` | `/api/dashboard` | **Required** | Personalised KPIs and analytics |
| `GET` | `/api/weather?city=X` | None | Weather by city name |
| `GET` | `/api/weather?lat=X&lon=Y` | None | Weather by GPS coordinates |
| `POST` | `/api/risk-analysis` | None | Standalone risk analysis |

---

## 🧪 Testing

Run the automated regression suite (requires server running):

```bash
# Start server in one terminal
python app.py

# Run regression tests in another
python verify_bugfix.py     # 29 API tests
python prove_scan_fix.py    # Scan persistence proof
```

---

## 🌿 Supported Diseases (38 Classes)

The model detects diseases across **14 plant species**:

| Plant | Diseases Detected |
|---|---|
| Tomato | Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Corn/Maize | Cercospora, Common Rust, Northern Blight, Healthy |
| Apple | Black Rot, Cedar Rust, Scab, Healthy |
| Grape | Black Rot, Esca, Isariopsis Leaf Spot, Healthy |
| + 9 more | Peach, Cherry, Pepper, Strawberry, Soybean, Squash, Blueberry, Raspberry, Orange |

---

## 🔒 Security Features

- **Password hashing**: `bcrypt` with salt rounds
- **JWT tokens**: HS256 signed, 24-hour expiry
- **Data isolation**: All MongoDB queries scoped by `userId` (ObjectId)
- **Optional auth on `/predict`**: JWT injected automatically when logged in
- **No secrets in source**: All credentials in `.env` (git-ignored)
- **CORS**: Configured for local dev; restrict origins in production

---

## 📸 Screenshots

> *Glassmorphic dark UI with real-time intelligence*

| Feature | Screenshot |
|---|---|
| Disease Detection | Upload → AI Analysis → Risk Card |
| Weather Dashboard | Live weather + Smart Farming Insights |
| Analytics Page | KPI cards + Risk distribution chart |
| Scan History | Per-user MongoDB-backed history with risk badges |

---

## 🗺️ Roadmap

- [x] **Phase 1A** — JWT Auth + Multi-User SaaS
- [x] **Phase 1B** — User Dashboard & Analytics
- [x] **Phase 1C** — Real-Time Weather Integration
- [x] **Phase 1D** — Farming Insights Engine
- [x] **Phase 1E** — Disease Risk Prediction Engine
- [ ] **Phase 2** — Subscription tiers + Stripe billing
- [ ] **Phase 2** — Mobile PWA with offline detection
- [ ] **Phase 3** — Multi-farm management + team accounts
- [ ] **Phase 3** — Historical weather trend analysis

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/phase-2-billing`)
3. Commit your changes (`git commit -m 'Add Stripe billing integration'`)
4. Push to the branch (`git push origin feature/phase-2-billing`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with 🌿 by the AgroAI Team · Powered by TensorFlow, Flask, MongoDB Atlas & Google Gemini

</div>
