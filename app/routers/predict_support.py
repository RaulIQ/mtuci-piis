from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import HTTPException

from app.dependencies import AppDependencies
from app.monitoring import REQUESTS_TOTAL

T = TypeVar("T")


def predict_route_guard(
    endpoint: str,
    deps: AppDependencies,
    log_event: str,
    run: Callable[[], T],
) -> T:
    try:
        return run()
    except ValueError as exc:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        deps.logger.exception("%s error=%s", log_event, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def predict_route_guard_async(
    endpoint: str,
    deps: AppDependencies,
    log_event: str,
    run: Callable[[], Awaitable[T]],
) -> T:
    try:
        return await run()
    except ValueError as exc:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="bad_request").inc()
        raise
    except Exception as exc:  # pylint: disable=broad-except
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        deps.logger.exception("%s error=%s", log_event, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
