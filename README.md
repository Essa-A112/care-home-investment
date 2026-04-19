# Care Home Investment Scoring Tool

A production-quality AI investment scoring platform for UK care home real estate. Every Local Authority District (LAD) in England is scored by investment potential using public government data and machine learning.

---

## Overview

The tool ingests publicly available datasets — demographic trends, care home supply, health need proxies, and deprivation indices — engineers features at the LAD level, trains a CatBoost gradient-boosted model, and exposes ranked investment scores via a FastAPI REST API with a React dashboard.

---

## Data Sources

| Source | Dataset | URL |
|--------|---------|-----|
| ONS | Population estimates & projections (65+) | https://www.ons.gov.uk |
| CQC | Care home registrations, beds, and ratings | https://www.cqc.org.uk/about-us/transparency/using-cqc-data |
| NHS England | Hospital admissions & delayed discharges | https://www.england.nhs.uk/statistics |
| MHCLG | Indices of Multiple Deprivation (IMD) | https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019 |
| ONS | LAD boundary files (GeoJSON) | https://geoportal.statistics.gov.uk |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI + Uvicorn |
| ML Model | CatBoost + SHAP explainability |
| Data processing | Pandas, NumPy, scikit-learn |
| Frontend | React (Vite) |
| Deployment | AWS EC2 (Docker) |
| Secrets | python-dotenv (.env) |
| Version control | GitHub |

---

## Project Structure

```
care-home-investment/
├── api/                    # FastAPI application
│   ├── main.py             # App entrypoint
│   ├── routes/             # Route handlers
│   └── models/             # Pydantic request/response schemas
├── data_ingestion/         # Raw data downloaders by source
│   └── sources/            # One module per data provider
├── data_processing/        # Feature engineering & merging
├── modelling/              # Model training, inference, SHAP
├── frontend/               # React dashboard (Vite)
├── tests/                  # Pytest test suite
├── data/                   # Local data cache (gitignored)
├── Dockerfile              # Backend container
├── requirements.txt
├── .env.example
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (frontend)
- Docker (optional, for container-based dev)

### Backend

```bash
git clone https://github.com/Essa-A112/care-home-investment.git
cd care-home-investment

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Fill in your secrets
uvicorn api.main:app --reload
```

API docs available at http://localhost:8000/docs

### Running Tests

```bash
pytest tests/
```

---

## Deployment (AWS EC2)

```bash
# On your EC2 instance
docker build -t care-home-investment .
docker run -d \
  --env-file .env \
  -p 8000:8000 \
  care-home-investment
```

Ensure your EC2 security group allows inbound traffic on port 8000 (or sit behind an Application Load Balancer on 443).

---

## Environment Variables

Copy `.env.example` to `.env` and populate:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GPT-powered narrative generation for reports |
| `AWS_ACCESS_KEY_ID` | S3 data storage access |
| `AWS_SECRET_ACCESS_KEY` | S3 data storage secret |
| `AWS_REGION` | AWS region (e.g. `eu-west-2`) |

---

## Licence

MIT
