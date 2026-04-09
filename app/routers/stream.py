import time

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies import AppDependencies, get_deps
from app.label_parsing import parse_comma_separated_labels
from app.routers.route_errors import run_route_guard
from app.schemas import StreamPredictResponse
from app.streaming import StreamParams
from app.use_cases.stream_from_bytes import run_stream_from_bytes
from app.use_cases.stream_from_logmel_npz import run_stream_from_logmel_npz

router = APIRouter(tags=["stream"])


@router.post("/predict-stream", response_model=StreamPredictResponse)
async def predict_stream(
    file: UploadFile = File(...),
    stride_sec: float = Form(0.25),
    refractory_sec: float = Form(0.8),
    confidence_threshold: float = Form(0.55),
    target_labels: str = Form("one,two,three,four,five,six,seven,eight,nine"),
    deps: AppDependencies = Depends(get_deps),
):
    endpoint = "/predict-stream"
    started = time.perf_counter()

    async def _run() -> StreamPredictResponse:
        content = await file.read()
        params = StreamParams(
            stride_sec=stride_sec,
            refractory_sec=refractory_sec,
            confidence_threshold=confidence_threshold,
            target_labels=parse_comma_separated_labels(target_labels),
        )
        return run_stream_from_bytes(
            streaming=deps.streaming,
            raw_bytes=content,
            filename=file.filename,
            endpoint=endpoint,
            started_perf=started,
            params=params,
            logger=deps.logger,
        )

    return await run_route_guard(endpoint, deps, "predict_stream_failed", _run)


@router.post("/predict-stream-logmel", response_model=StreamPredictResponse)
async def predict_stream_logmel(
    file: UploadFile = File(...),
    stride_sec: float = Form(0.25),
    refractory_sec: float = Form(0.8),
    confidence_threshold: float = Form(0.55),
    target_labels: str = Form("one,two,three,four,five,six,seven,eight,nine"),
    deps: AppDependencies = Depends(get_deps),
):
    endpoint = "/predict-stream-logmel"
    started = time.perf_counter()

    async def _run() -> StreamPredictResponse:
        content = await file.read()
        params = StreamParams(
            stride_sec=stride_sec,
            refractory_sec=refractory_sec,
            confidence_threshold=confidence_threshold,
            target_labels=parse_comma_separated_labels(target_labels),
        )
        return run_stream_from_logmel_npz(
            service=deps.service,
            raw_bytes=content,
            filename=file.filename,
            endpoint=endpoint,
            started_perf=started,
            params=params,
            logger=deps.logger,
            empty_detail="Empty file",
        )

    return await run_route_guard(endpoint, deps, "predict_stream_logmel_failed", _run)
