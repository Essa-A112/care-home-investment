from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Care Home Investment Scoring API",
    description="Scores every Local Authority District in England by care home investment potential.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
