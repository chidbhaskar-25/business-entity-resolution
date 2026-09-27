"""FastAPI backend application for Business Entity Resolution."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routes import router

app = FastAPI(
    title="Amazon ML Challenge 2026 - Business Entity Resolution API",
    description=(
        "Production-grade FastAPI backend orchestrating the ML pipeline for multi-source business entity resolution. "
        "Supports data inspection, model training, entity-grouped validation, threshold optimization (Macro F0.5), "
        "country-partitioned test inference, and strict submission validation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount pipeline API routes
app.include_router(router)
app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
def root():
    return {
        "title": "Amazon ML Challenge 2026: Business Entity Resolution API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "GET /health",
            "GET /dataset/stats",
            "POST /train",
            "POST /evaluate",
            "POST /predict",
            "POST /validate",
            "GET /metrics",
            "GET /experiments",
            "GET /matches/{source1_entity_id}",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
