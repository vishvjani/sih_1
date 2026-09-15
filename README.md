# SignalScope — AI Media Authenticity Detection Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-purple.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Smart India Hackathon (SIH 2026) | Problem Statement 2**  
Domain: **AI / Media Forensics / Trust & Safety**

SignalScope is a high-performance media authenticity detection system designed to identify whether an image is **authentic/real** or **AI-generated/synthetic**. It provides localized visual explanations, generator attribution, C2PA & EXIF metadata inspection, and degradation robustness testing.

---

## 🏗 Repository Structure

The project is structured into two clean, self-contained components:

```text
sih/
├── backend/                       # FastAPI Backend API Server
│   ├── src/app/                   # Application core
│   │   ├── api/v1/                # REST endpoints (/analyze, /metadata, /robustness, etc.)
│   │   ├── services/              # Forensic inference, calibration, explainability, attribution
│   │   ├── schemas/               # Pydantic request/response models
│   │   └── config.py              # Application settings
│   ├── tests/                     # Integration tests (pytest)
│   ├── requirements.txt           # Lightweight backend dependencies
│   └── README.md                  # Backend API documentation
│
└── frontend/                      # Modern React + Vite + Tailwind CSS Web Studio
    ├── src/                       # React components, 3D visualizers, API client
    ├── package.json               # Frontend dependencies & scripts
    └── vite.config.ts             # Vite configuration
```

---

## 🚀 Quick Start Guide

### 1. Start the Backend API (FastAPI)

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the server with hot-reload
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Docs (ReDoc)**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **Video : https://drive.google.com/file/d/1OwPvJ1oC581Aj6_Dbf1qOn_MqMDeSaOP/view?usp=drivesdk

### 2. Start the Frontend (React + Vite)

```bash
# Navigate to frontend directory
cd frontend

# Install node dependencies
npm install

# Start development server
npm run dev
```

- **Web Application**: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Running Backend Tests

Run the backend test suite using `pytest`:

```bash
cd backend
pytest tests/ -v
```

All 8 endpoint tests run in under a second and verify:
- Root and health status check
- Single image authenticity analysis
- Batch image processing
- EXIF and C2PA metadata inspection
- Image degradation robustness testing
- Multimodal image-text consistency verification
- Benchmark evaluation metrics

---

## 💻 Building Frontend for Production

```bash
cd frontend
npm run build
```

The production assets will be output to `frontend/dist/`.

---

## ⚖️ Ethics & Responsible AI Disclosure

SignalScope operates strictly as a **decision-support forensic system**. It communicates results using responsible likelihood terminology ("Likely AI-generated") and confidence intervals rather than absolute legal assertions. The system operates solely on image-level visual artifacts and is not intended for profiling identifiable individuals or adjudicating political claims.

