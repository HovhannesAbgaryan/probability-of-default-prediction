"""Prometheus metrics. Run one process per container (scale with replicas) so counters stay consistent."""
from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "route"],
                         buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PREDICTIONS = Counter("pd_predictions_total", "Scored applications", ["model_version", "grade"])
PD_VALUE = Histogram("pd_predicted_probability", "Predicted PD distribution (drift monitoring)", ["model_version"],
                     buckets=(0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.4, 0.6, 0.8, 1.0))
INPUT_WARNINGS = Counter("pd_input_warnings_total", "Inputs adjusted during preprocessing", ["code"])
