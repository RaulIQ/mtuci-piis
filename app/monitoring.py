import time

from prometheus_client import Counter, Histogram


REQUESTS_TOTAL = Counter(
    "kws_requests_total",
    "Total number of inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY_MS = Histogram(
    "kws_request_latency_ms",
    "Request latency in milliseconds",
    ["endpoint"],
    buckets=(5, 10, 25, 50, 100, 200, 500, 1000, 2000),
)

INFERENCE_LATENCY_MS = Histogram(
    "kws_inference_latency_ms",
    "Pure model inference latency in milliseconds",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)


def observe_request_success_ms(endpoint: str, started_perf: float) -> float:
    latency_ms = (time.perf_counter() - started_perf) * 1000
    REQUEST_LATENCY_MS.labels(endpoint=endpoint).observe(latency_ms)
    REQUESTS_TOTAL.labels(endpoint=endpoint, status="ok").inc()
    return latency_ms

