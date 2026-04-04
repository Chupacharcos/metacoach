"""MetaCoach — FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router

app = FastAPI(
    title="MetaCoach API",
    description="Entrenador personal con IA fisiológica — LangChain + Groq + LLaMA 3.1 70B",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "MetaCoach",
        "version": "1.0.0",
        "endpoints": ["/metacoach/generate-plan", "/metacoach/chat", "/metacoach/health"],
    }
