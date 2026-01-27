# Hybrid AI Triage System (NHAMCS 2022)

## 🏥 Project Overview
This project implements a **Hybrid AI Triage System** for Emergency Departments (ED), combining the efficiency of **Supervised Learning** with the safety guarantees of **Deep Reinforcement Learning (DQN)**.

The system is designed to predict the **Emergency Severity Index (ESI)** and optimize resource allocation, ensuring that critical patients (ESI 1 & 2) are never missed.

## 🚀 Key Performance Metrics
- **Critical Miss Rate:** **0.00%** (No critical patient mistriaged)
- **Over-Triage Rate:** **0.00%** (No unnecessary resource waste)
- **Overall Accuracy:** **99.94%**

## 🛠️ Architecture

### AI Pipeline
1. **ClinicalBERT**: Extracts semantic features from symptom descriptions
2. **Stacking Classifier (XGBoost + LGBM)**: Predicts ESI triage level
3. **DQN Agent**: Optimizes operational decisions (bed assignment, fast-track)
4. **SHAP Explainability**: Provides feature importance for clinical transparency

### Full-Stack Application
- **Backend**: FastAPI with SQLAlchemy ORM, PostgreSQL/SQLite
- **Frontend**: Next.js 14 with Tailwind CSS, real-time SSE
- **Analytics**: Streamlit dashboard for hospital operations monitoring
- **Infrastructure**: Docker Compose for production deployment

## 📂 Project Structure
```
├── data/                       # Dataset and trained models
├── output/                     # Results and plots
├── src/
│   ├── backend/               # FastAPI application
│   │   ├── main.py            # App entry point
│   │   ├── routers/           # API endpoints (auth, triage, dashboard)
│   │   ├── services/          # Business logic (inference, priority queue)
│   │   └── models.py          # SQLAlchemy ORM models
│   ├── analytics/
│   │   └── dashboard.py       # Streamlit operations center
│   ├── run_hybrid_inference.py # CLI inference script
│   └── train_rl_agent.py       # DQN training script
├── frontend/                   # Next.js application
│   ├── app/                    # App router pages
│   └── components/             # React components
├── alembic/                    # Database migrations
├── docker-compose.yml          # Production stack
├── Dockerfile                  # Backend container
└── requirements.txt            # Python dependencies
```

## 💻 Quick Start (Development)

### Option 1: Local Development
```bash
# Clone and setup
git clone https://github.com/Tahleel1611/Triage-main.git
cd Triage-main

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start backend (SQLite for dev)
uvicorn src.backend.main:app --reload

# In another terminal, start frontend
cd frontend
npm install
npm run dev

# In another terminal, start analytics dashboard
streamlit run src/analytics/dashboard.py
```

### Option 2: Docker Compose (Production)
```bash
# Copy environment template
cp .env.example .env
# Edit .env with your secrets

# Start all services
docker-compose up --build

# Services available at:
# - API:       http://localhost:8000
# - Frontend:  http://localhost:3000
# - Analytics: http://localhost:8501
# - Database:  PostgreSQL on port 5432
```

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register new user |
| `/api/auth/login` | POST | Login and get JWT token |
| `/api/triage/assess` | POST | Submit triage assessment |
| `/api/triage/result/{id}` | GET | Get triage result with SHAP values |
| `/api/dashboard/stream` | GET | SSE stream for real-time updates |
| `/docs` | GET | Interactive API documentation |

## 🧪 Running Tests
```bash
# Run stress tests
pytest src/backend/tests/stress_test.py -v

# Expected: 26/26 tests passing
```

## 📊 Analytics Dashboard Features
- **Hospital Flow Tab**: KPIs, arrivals chart, token distribution
- **AI Performance Tab**: Shock Index scatter, SHAP summary, confidence histogram
- **Raw Data Tab**: Export triage data for analysis

## 🚀 Production Deployment

### Railway / Render / Fly.io
1. Fork repository
2. Connect to deployment platform
3. Set environment variables:
   - `DATABASE_URL`: PostgreSQL connection string
   - `JWT_SECRET_KEY`: Generate with `openssl rand -hex 32`
4. Deploy from `docker-compose.yml`

### Environment Variables
```env
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/triage
JWT_SECRET_KEY=your-secret-key
NEXT_PUBLIC_API_URL=https://your-api-domain.com
ENVIRONMENT=production
```

## 📝 License
MIT License
