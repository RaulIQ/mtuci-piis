from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.dependencies import AppDependencies, get_deps

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(deps: AppDependencies = Depends(get_deps)):
    return {
        "status": "ready",
        "model_version": deps.service.model_version,
        "labels": deps.service.labels,
    }


@router.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
