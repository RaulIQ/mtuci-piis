import inspect
from collections.abc import Awaitable, Callable
from typing import NoReturn, TypeVar

from fastapi import HTTPException

from app.dependencies import AppDependencies
from app.monitoring import REQUESTS_TOTAL

T = TypeVar("T")


async def run_route_guard(
    endpoint: str,
    deps: AppDependencies,
    log_event: str,
    run: Callable[[], T | Awaitable[T]],
) -> T:
    try:
        out = run()
        if inspect.isawaitable(out):
            out = await out
        return out  # type: ignore[no-any-return]
    except Exception as exc:  # pylint: disable=broad-except
        handle_route_exception(endpoint, deps, log_event, exc)


def handle_route_exception(
    endpoint: str,
    deps: AppDependencies,
    log_event: str,
    exc: Exception,
) -> NoReturn:
    if isinstance(exc, ValueError):
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, HTTPException):
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise
    REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
    deps.logger.exception("%s error=%s", log_event, exc)
    raise HTTPException(status_code=500, detail=str(exc)) from exc
