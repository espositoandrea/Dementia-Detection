import os
from typing import Callable, Tuple

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")


instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# Custom metrics


def model_output(
    metric_name: str = "model_output",
    metric_doc: str = "Output value of the model",
    metric_namespace: str = "",
    metric_subsystem: str = "",
    buckets: Tuple[int] = (0, .25, .50, .75, 1),
) -> Callable[[Info], None]:
    metric = Histogram(
        metric_name,
        metric_doc,
        labelnames=["endpoint"],
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler in ["/report", "/predict"]:
            predicted_probability = float(
                info.response.headers['X-predicted-probability'])
            if predicted_probability:
                metric.labels(info.modified_handler[1:]).observe(
                    predicted_probability)
    return instrumentation


def output_format(
    metric_name: str = "output_format",
    metric_doc: str = "Format chosen for the output",
    metric_namespace: str = "",
    metric_subsystem: str = "",
) -> Callable[[Info], None]:
    metric = Counter(
        metric_name,
        metric_doc,
        labelnames=["format", "endpoint"],
        namespace=metric_namespace,
        subsystem=metric_subsystem
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler in ["/report", "/predict"]:
            format = info.request.query_params.get("format", "json")
            metric.labels(format, info.modified_handler).inc()
    return instrumentation


instrumentator.add(
    model_output(
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM
    )
)
instrumentator.add(
    output_format(
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM
    )
)
