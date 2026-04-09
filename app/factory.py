import logging

from fastapi import FastAPI

from app.dependencies import AppDependencies
from app.model import KwsInferenceService
from app.routers import health, predict, stream, ws
from app.settings import DB_PATH, MODEL_PATH
from app.storage import RequestLogStore
from app.streaming import SlidingWindowProcessor


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("kws-service")

    store = RequestLogStore(DB_PATH)
    service = KwsInferenceService(MODEL_PATH)
    streaming = SlidingWindowProcessor(service)

    app = FastAPI(title="KWS Inference Service", version="1.0.0")
    app.state.deps = AppDependencies(
        store=store,
        service=service,
        streaming=streaming,
        logger=logger,
    )

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(stream.router)
    app.include_router(ws.router)

    return app
