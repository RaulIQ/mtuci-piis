import base64
import time

from fastapi import APIRouter, Depends, File, UploadFile

from app.dependencies import AppDependencies, get_deps
from app.routers.route_errors import run_route_guard
from app.schemas import KwsMelConfigResponse, PredictAudioRequest, PredictResponse
from app.use_cases.predict_from_bytes import run_predict_from_audio_bytes
from app.use_cases.predict_from_logmel_npy import run_predict_from_logmel_npy

router = APIRouter(tags=["predict"])


@router.get("/model/mel-config", response_model=KwsMelConfigResponse)
def mel_config(deps: AppDependencies = Depends(get_deps)):
    return KwsMelConfigResponse(**deps.service.mel_frontend_config())


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    deps: AppDependencies = Depends(get_deps),
):
    endpoint = "/predict"
    started = time.perf_counter()

    async def _run() -> PredictResponse:
        content = await file.read()
        return run_predict_from_audio_bytes(
            service=deps.service,
            store=deps.store,
            raw_bytes=content,
            filename=file.filename,
            endpoint=endpoint,
            started_perf=started,
            logger=deps.logger,
            empty_detail="Empty file",
        )

    return await run_route_guard(endpoint, deps, "predict_failed", _run)


@router.post("/predict-logmel", response_model=PredictResponse)
async def predict_logmel(
    file: UploadFile = File(...),
    deps: AppDependencies = Depends(get_deps),
):
    endpoint = "/predict-logmel"
    started = time.perf_counter()

    async def _run() -> PredictResponse:
        content = await file.read()
        return run_predict_from_logmel_npy(
            service=deps.service,
            store=deps.store,
            raw_bytes=content,
            filename=file.filename,
            endpoint=endpoint,
            started_perf=started,
            logger=deps.logger,
            empty_detail="Empty file",
        )

    return await run_route_guard(endpoint, deps, "predict_logmel_failed", _run)


@router.post("/predict-base64", response_model=PredictResponse)
async def predict_base64(
    payload: PredictAudioRequest,
    deps: AppDependencies = Depends(get_deps),
):
    endpoint = "/predict-base64"
    started = time.perf_counter()

    def _run() -> PredictResponse:
        content = base64.b64decode(payload.audio_base64)
        return run_predict_from_audio_bytes(
            service=deps.service,
            store=deps.store,
            raw_bytes=content,
            filename="base64_payload.wav",
            endpoint=endpoint,
            started_perf=started,
            logger=deps.logger,
            empty_detail="Empty payload",
            log_success=False,
        )

    return await run_route_guard(endpoint, deps, "predict_base64_failed", _run)
