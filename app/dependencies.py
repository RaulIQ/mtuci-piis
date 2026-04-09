import logging
from dataclasses import dataclass

from starlette.requests import HTTPConnection

from app.model import KwsInferenceService
from app.storage import RequestLogStore
from app.streaming import SlidingWindowProcessor


@dataclass(frozen=True)
class AppDependencies:
    store: RequestLogStore
    service: KwsInferenceService
    streaming: SlidingWindowProcessor
    logger: logging.Logger


def get_deps(conn: HTTPConnection) -> AppDependencies:
    return conn.app.state.deps
