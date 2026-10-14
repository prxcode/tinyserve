import time
from collections.abc import Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from tinyserve.sequence import FinishReason

REQUEST_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)
TOKEN_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


class Metrics:
    """Prometheus metrics for one server. Each instance has its own registry so tests can build
    several apps in one process without name clashes."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        registry = self.registry
        self.time_to_first_token = Histogram(
            "tinyserve_time_to_first_token_seconds",
            "Time from a request arriving to its first generated token.",
            buckets=REQUEST_BUCKETS,
            registry=registry,
        )
        self.inter_token_latency = Histogram(
            "tinyserve_inter_token_latency_seconds",
            "Time between consecutive generated tokens of one request.",
            buckets=TOKEN_BUCKETS,
            registry=registry,
        )
        self.request_latency = Histogram(
            "tinyserve_request_latency_seconds",
            "Time from a request arriving to its final token.",
            buckets=REQUEST_BUCKETS,
            registry=registry,
        )
        self.requests = Counter(
            "tinyserve_requests",
            "Finished requests by finish reason.",
            ["finish_reason"],
            registry=registry,
        )
        self.rejected = Counter(
            "tinyserve_rejected_requests",
            "Requests rejected before reaching the engine.",
            ["reason"],
            registry=registry,
        )
        self.prompt_tokens = Counter(
            "tinyserve_prompt_tokens", "Prompt tokens of accepted requests.", registry=registry
        )
        self.generated_tokens = Counter(
            "tinyserve_generated_tokens", "Tokens generated for all requests.", registry=registry
        )
        self.step_duration = Histogram(
            "tinyserve_engine_step_seconds",
            "Duration of one engine step, by step kind.",
            ["kind"],
            buckets=TOKEN_BUCKETS,
            registry=registry,
        )
        self.batch_size = Histogram(
            "tinyserve_engine_batch_size",
            "Sequences in one engine step, by step kind.",
            ["kind"],
            buckets=BATCH_BUCKETS,
            registry=registry,
        )
        self.preemptions = Counter(
            "tinyserve_preemptions",
            "Sequences preempted because the KV cache ran out of blocks.",
            registry=registry,
        )

    def watch(
        self,
        running: Callable[[], float],
        waiting: Callable[[], float],
        pending: Callable[[], float],
        kv_cache_usage: Callable[[], float],
    ) -> None:
        """Register gauges that are read from the engine when Prometheus scrapes."""
        gauges = [
            ("tinyserve_running_requests", "Sequences currently being decoded.", running),
            ("tinyserve_waiting_requests", "Sequences waiting for admission.", waiting),
            ("tinyserve_pending_requests", "Requests accepted but not finished.", pending),
            (
                "tinyserve_kv_cache_usage_ratio",
                "Fraction of KV cache blocks in use.",
                kv_cache_usage,
            ),
        ]
        for name, description, read in gauges:
            Gauge(name, description, registry=self.registry).set_function(read)

    def observe_step(self, is_prefill: bool, batch_size: int, seconds: float) -> None:
        kind = "prefill" if is_prefill else "decode"
        self.step_duration.labels(kind).observe(seconds)
        self.batch_size.labels(kind).observe(batch_size)


class RequestTimer:
    """Records the latency metrics of one request as its tokens reach the HTTP layer."""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._start = time.perf_counter()
        self._last_token: float | None = None

    def token(self) -> None:
        now = time.perf_counter()
        if self._last_token is None:
            self._metrics.time_to_first_token.observe(now - self._start)
        else:
            self._metrics.inter_token_latency.observe(now - self._last_token)
        self._last_token = now
        self._metrics.generated_tokens.inc()

    def finish(self, reason: FinishReason) -> None:
        self._metrics.request_latency.observe(time.perf_counter() - self._start)
        self._metrics.requests.labels(reason.value).inc()
