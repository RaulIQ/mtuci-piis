import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import AppDependencies, get_deps
from app.monitoring import REQUESTS_TOTAL
from app.schemas import StreamPredictResponse
from app.streaming import StreamParams
from app.use_cases.stream_from_bytes import run_stream_from_bytes

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
    try:
        content = await file.read()
        labels_set = {x.strip() for x in target_labels.split(",") if x.strip()}
        params = StreamParams(
            stride_sec=stride_sec,
            refractory_sec=refractory_sec,
            confidence_threshold=confidence_threshold,
            target_labels=labels_set,
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
    except ValueError as exc:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        deps.logger.exception("predict_stream_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
