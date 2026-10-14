"""Behavioural tests for the token-bucket rate limiter.

Time is fully controlled: ``time.time`` inside the rate limiter module is replaced
by a fake clock, and ``asyncio.sleep`` advances that clock instead of blocking, so
every assertion on tokens, backoff and wait times is exact and deterministic.
"""

import asyncio
import types
from typing import Any, Dict

import pytest

import dataquery.transport.rate_limiter as rl_module
from dataquery import constants as C
from dataquery.transport.rate_limiter import (
    EnhancedTokenBucketRateLimiter,
    QueuePriority,
    RateLimitConfig,
    RateLimitContext,
    TokenBucketRateLimiter,
    create_rate_limiter,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(rl_module, "time", types.SimpleNamespace(time=fake.time))
    return fake


@pytest.fixture
def sleeps(monkeypatch, clock):
    """Record every sleep the limiter requests and advance the fake clock by it."""
    real_sleep = asyncio.sleep
    recorded = []

    async def fake_sleep(delay):
        recorded.append(delay)
        clock.advance(delay)
        await real_sleep(0)

    monkeypatch.setattr(rl_module.asyncio, "sleep", fake_sleep)
    return recorded


@pytest.fixture
def make_limiter(clock):
    def _make(**overrides) -> EnhancedTokenBucketRateLimiter:
        params: Dict[str, Any] = {"requests_per_minute": 60, "burst_capacity": 5}
        params.update(overrides)
        limiter = EnhancedTokenBucketRateLimiter(RateLimitConfig(**params))
        # RateLimitState's dataclass default factories bind the real time.time.
        limiter.state.window_start = clock.now
        limiter.state.last_request_time = clock.now
        return limiter

    return _make


class TestRateLimitConfigValidation:
    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"requests_per_minute": 0}, "requests_per_minute must be positive"),
            ({"requests_per_minute": -5}, "requests_per_minute must be positive"),
            ({"burst_capacity": 0}, "burst_capacity must be positive"),
            ({"window_size_seconds": 0}, "window_size_seconds must be positive"),
            ({"requests_per_minute": 10, "burst_capacity": 11}, "burst_capacity cannot exceed requests_per_minute"),
            ({"max_queue_size": 0}, "max_queue_size must be positive"),
            ({"queue_timeout": 0}, "queue_timeout must be positive"),
            ({"queue_timeout": -1.0}, "queue_timeout must be positive"),
        ],
    )
    def test_invalid_values_are_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            RateLimitConfig(**overrides)

    def test_burst_equal_to_requests_per_minute_is_allowed(self):
        cfg = RateLimitConfig(requests_per_minute=10, burst_capacity=10)
        assert cfg.burst_capacity == 10

    def test_new_limiter_starts_with_full_bucket(self, make_limiter, clock):
        limiter = make_limiter(burst_capacity=4)
        assert limiter.state.tokens == 4.0
        assert limiter.state.last_refill == clock.now


class TestTokenRefill:
    def test_refills_at_requests_per_minute_rate(self, make_limiter, clock):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=5)
        limiter.state.tokens = 0.0

        clock.advance(2.5)
        limiter._refill_tokens()

        assert limiter.state.tokens == pytest.approx(2.5)
        assert limiter.state.last_refill == clock.now

    def test_refill_never_exceeds_burst_capacity(self, make_limiter, clock):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=5)
        limiter.state.tokens = 1.0

        clock.advance(1_000)
        limiter._refill_tokens()

        assert limiter.state.tokens == 5.0

    def test_adaptive_backoff_caps_elapsed_time_credited(self, make_limiter, clock):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=5)
        limiter.state.tokens = 0.0
        limiter.state.current_backoff = 2.0

        clock.advance(10)
        limiter._refill_tokens()

        # Only 2s (the backoff) of the 10s elapsed is credited at 1 token/s.
        assert limiter.state.tokens == pytest.approx(2.0)

    def test_backoff_is_ignored_when_adaptive_limiting_disabled(self, make_limiter, clock):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=5, adaptive_rate_limiting=False)
        limiter.state.tokens = 0.0
        limiter.state.current_backoff = 2.0

        clock.advance(10)
        limiter._refill_tokens()

        assert limiter.state.tokens == 5.0

    def test_request_counter_resets_when_window_elapses(self, make_limiter, clock):
        limiter = make_limiter(window_size_seconds=60)
        window_start = limiter.state.window_start
        limiter.state.request_count = 7

        clock.advance(59.9)
        limiter._refill_tokens()
        assert limiter.state.request_count == 7
        assert limiter.state.window_start == window_start

        clock.advance(0.1)
        limiter._refill_tokens()
        assert limiter.state.request_count == 0
        assert limiter.state.window_start == clock.now


class TestWaitTimeCalculation:
    def test_wait_is_one_token_interval(self, make_limiter):
        assert make_limiter(requests_per_minute=60)._calculate_wait_time() == pytest.approx(1.0)
        assert make_limiter(requests_per_minute=30, burst_capacity=5)._calculate_wait_time() == pytest.approx(2.0)

    def test_wait_is_floored_at_minimum(self, make_limiter):
        # 600 rpm -> 0.1s interval, below the configured floor.
        limiter = make_limiter(requests_per_minute=600)
        assert limiter._calculate_wait_time() == C.RATE_LIMIT_MIN_WAIT_SECONDS

    def test_active_backoff_extends_wait(self, make_limiter):
        limiter = make_limiter(requests_per_minute=60)
        limiter.state.current_backoff = 5.0
        assert limiter._calculate_wait_time() == 5.0

    def test_backoff_shorter_than_interval_does_not_shorten_wait(self, make_limiter):
        limiter = make_limiter(requests_per_minute=60)
        limiter.state.current_backoff = 0.5
        assert limiter._calculate_wait_time() == pytest.approx(1.0)

    def test_backoff_ignored_when_adaptive_disabled(self, make_limiter):
        limiter = make_limiter(requests_per_minute=60, adaptive_rate_limiting=False)
        limiter.state.current_backoff = 5.0
        assert limiter._calculate_wait_time() == pytest.approx(1.0)


class TestServerRateLimitResponses:
    @pytest.mark.parametrize(
        "rpm, expected_first, expected_second",
        [
            (60, 1.5, 2.25),  # 1s interval -> base 1.0
            (300, 1.5, 2.25),  # 0.2s interval -> base floored at 1.0
            (30, 3.0, 4.5),  # 2s interval -> base 2.0
        ],
    )
    def test_backoff_seeds_from_base_and_grows_geometrically(
        self, make_limiter, clock, rpm, expected_first, expected_second
    ):
        limiter = make_limiter(requests_per_minute=rpm, burst_capacity=5)

        limiter.handle_rate_limit_response({"Retry-After": "7"})
        assert limiter.state.current_backoff == pytest.approx(expected_first)
        assert limiter.state.retry_after == 7.0
        assert limiter.state.consecutive_failures == 1
        assert limiter.state.last_failure_time == clock.now

        clock.advance(3)
        limiter.handle_rate_limit_response({"Retry-After": "2.5"})
        assert limiter.state.current_backoff == pytest.approx(expected_second)
        assert limiter.state.retry_after == 2.5
        assert limiter.state.consecutive_failures == 2
        assert limiter.state.last_failure_time == clock.now

    def test_backoff_is_capped_at_max_backoff(self, make_limiter):
        limiter = make_limiter()
        limiter.config.max_backoff_seconds = 10.0

        for _ in range(20):
            limiter.handle_rate_limit_response({"Retry-After": "1"})

        assert limiter.state.current_backoff == 10.0
        assert limiter.state.consecutive_failures == 20

    @pytest.mark.parametrize("headers", [{}, {"Retry-After": ""}, {"Other": "5"}])
    def test_missing_retry_after_header_changes_nothing(self, make_limiter, headers):
        limiter = make_limiter()

        limiter.handle_rate_limit_response(headers)

        assert limiter.state.retry_after is None
        assert limiter.state.consecutive_failures == 0
        assert limiter.state.current_backoff == 0.0

    @pytest.mark.parametrize("value", ["soon", "Wed, 21 Oct 2026 07:28:00 GMT", "1,5"])
    def test_unparseable_retry_after_is_ignored(self, make_limiter, value):
        limiter = make_limiter()

        limiter.handle_rate_limit_response({"Retry-After": value})

        assert limiter.state.retry_after is None
        assert limiter.state.consecutive_failures == 0
        assert limiter.state.current_backoff == 0.0

    def test_custom_retry_after_header_name(self, make_limiter):
        limiter = make_limiter(retry_after_header="X-RateLimit-Reset")

        limiter.handle_rate_limit_response({"Retry-After": "9"})
        assert limiter.state.consecutive_failures == 0

        limiter.handle_rate_limit_response({"X-RateLimit-Reset": "4"})
        assert limiter.state.retry_after == 4.0
        assert limiter.state.consecutive_failures == 1

    def test_non_adaptive_records_failure_without_backoff(self, make_limiter):
        limiter = make_limiter(adaptive_rate_limiting=False)

        limiter.handle_rate_limit_response({"Retry-After": "3"})

        assert limiter.state.retry_after == 3.0
        assert limiter.state.consecutive_failures == 1
        assert limiter.state.current_backoff == 0.0

    def test_success_clears_adaptive_backoff(self, make_limiter):
        limiter = make_limiter()
        limiter.handle_rate_limit_response({"Retry-After": "1"})
        limiter.handle_rate_limit_response({"Retry-After": "1"})

        limiter.handle_successful_request()

        assert limiter.state.consecutive_failures == 0
        assert limiter.state.current_backoff == 0.0
        assert limiter._calculate_wait_time() == pytest.approx(1.0)

    def test_success_does_not_clear_failures_when_adaptive_disabled(self, make_limiter):
        limiter = make_limiter(adaptive_rate_limiting=False)
        limiter.handle_rate_limit_response({"Retry-After": "1"})

        limiter.handle_successful_request()

        assert limiter.state.consecutive_failures == 1

    def test_stats_reflect_rate_limit_state(self, make_limiter):
        limiter = make_limiter()
        limiter.handle_rate_limit_response({"Retry-After": "4"})

        stats = limiter.get_stats()

        assert stats["retry_after"] == 4.0
        assert stats["consecutive_failures"] == 1
        assert stats["current_backoff"] == pytest.approx(1.5)
        assert stats["tokens_available"] == 5.0
        assert stats["burst_capacity"] == 5
        assert stats["requests_per_minute"] == 60


class TestAcquire:
    @pytest.mark.asyncio
    async def test_disabled_limiter_always_grants_without_spending(self, make_limiter, sleeps):
        limiter = make_limiter(enable_rate_limiting=False)
        limiter.state.tokens = 0.0

        results = [await limiter.acquire(timeout=0.0) for _ in range(10)]

        assert all(results)
        assert limiter.state.tokens == 0.0
        assert limiter.state.request_count == 0
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_grant_spends_token_and_records_request(self, make_limiter, clock, sleeps):
        limiter = make_limiter(burst_capacity=3)

        assert await limiter.acquire() is True

        assert limiter.state.tokens == pytest.approx(2.0)
        assert limiter.state.request_count == 1
        assert limiter.state.last_request_time == clock.now
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_burst_is_granted_immediately_then_throttled(self, make_limiter, sleeps):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=3)

        burst = [await limiter.acquire(timeout=0.5) for _ in range(3)]
        throttled = await limiter.acquire(timeout=0.5)

        assert burst == [True, True, True]
        assert throttled is False
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_without_timeout_waits_until_a_token_refills(self, make_limiter, clock, sleeps):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=1)
        assert await limiter.acquire()
        start = clock.now

        assert await limiter.acquire(timeout=None) is True

        # Polls in bounded steps rather than one long sleep, and resumes once refilled.
        assert sleeps
        assert all(d <= C.RATE_LIMIT_POLL_INTERVAL_SECONDS for d in sleeps)
        assert clock.now - start == pytest.approx(1.0, abs=C.RATE_LIMIT_POLL_INTERVAL_SECONDS)
        assert limiter.state.request_count == 2
        assert limiter.state.tokens < 1.0

    @pytest.mark.asyncio
    async def test_timeout_shorter_than_wait_fails_fast_without_sleeping(self, make_limiter, sleeps):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=1)
        assert await limiter.acquire()

        assert await limiter.acquire(timeout=0.5) is False

        assert sleeps == []
        assert limiter.state.request_count == 1

    @pytest.mark.asyncio
    async def test_generous_timeout_waits_and_succeeds(self, make_limiter, clock, sleeps):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=1)
        assert await limiter.acquire()
        start = clock.now

        assert await limiter.acquire(timeout=5.0) is True

        assert clock.now - start == pytest.approx(1.0, abs=C.RATE_LIMIT_POLL_INTERVAL_SECONDS)

    @pytest.mark.asyncio
    async def test_server_backoff_makes_acquire_give_up_sooner(self, make_limiter, sleeps):
        no_backoff = make_limiter(requests_per_minute=60, burst_capacity=1)
        assert await no_backoff.acquire()
        await no_backoff.acquire(timeout=1.2)
        assert sleeps, "a 1.2s budget covers the 1s interval, so the caller should wait"

        sleeps.clear()
        backed_off = make_limiter(requests_per_minute=60, burst_capacity=1)
        assert await backed_off.acquire()
        backed_off.handle_rate_limit_response({"Retry-After": "1"})  # backoff -> 1.5s

        assert await backed_off.acquire(timeout=1.2) is False
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_backoff_slows_refill_while_waiting(self, make_limiter, clock, sleeps):
        limiter = make_limiter(requests_per_minute=60, burst_capacity=1)
        assert await limiter.acquire()
        limiter.state.current_backoff = 0.05  # each poll credits at most 0.05s

        start = clock.now
        assert await limiter.acquire(timeout=None) is True

        # 0.1s polls credit only 0.05s each, so a 1-token refill takes ~2s, not ~1s.
        assert clock.now - start == pytest.approx(2.0, abs=C.RATE_LIMIT_POLL_INTERVAL_SECONDS)


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_restores_fresh_state(self, make_limiter, clock):
        limiter = make_limiter(burst_capacity=3)
        for _ in range(3):
            await limiter.acquire()
        limiter.handle_rate_limit_response({"Retry-After": "5"})
        clock.advance(1.0)

        limiter.reset()
        await asyncio.sleep(0)

        assert limiter.state.tokens == 3.0
        assert limiter.state.request_count == 0
        assert limiter.state.consecutive_failures == 0
        assert limiter.state.current_backoff == 0.0
        assert limiter.state.retry_after is None
        assert limiter.state.last_refill == clock.now

    @pytest.mark.asyncio
    async def test_reset_waits_for_the_bucket_lock(self, make_limiter):
        limiter = make_limiter()
        limiter.state.consecutive_failures = 3
        lock = limiter._get_lock()

        await lock.acquire()
        try:
            limiter.reset()
            await asyncio.sleep(0)
            assert limiter.state.consecutive_failures == 3
        finally:
            lock.release()

        for _ in range(3):
            await asyncio.sleep(0)
        assert limiter.state.consecutive_failures == 0

    def test_reset_requires_a_running_event_loop(self, make_limiter):
        limiter = make_limiter()
        with pytest.raises(RuntimeError):
            limiter.reset()


class TestRateLimitContextBehaviour:
    @pytest.mark.asyncio
    async def test_forwards_timeout_priority_and_operation(self, make_limiter):
        limiter = make_limiter()
        calls = []

        async def fake_acquire(timeout, priority, operation):
            calls.append((timeout, priority, operation))
            return True

        limiter.acquire = fake_acquire

        async with RateLimitContext(limiter, timeout=0.5, priority=QueuePriority.HIGH, operation="GET_files") as ctx:
            assert ctx.acquired is True

        assert calls == [(0.5, QueuePriority.HIGH, "GET_files")]

    @pytest.mark.asyncio
    async def test_successful_block_clears_backoff(self, make_limiter, sleeps):
        limiter = make_limiter()
        limiter.handle_rate_limit_response({"Retry-After": "1"})

        async with RateLimitContext(limiter, timeout=1.0):
            pass

        assert limiter.state.consecutive_failures == 0
        assert limiter.state.current_backoff == 0.0

    @pytest.mark.asyncio
    async def test_failing_block_keeps_backoff_and_propagates(self, make_limiter, sleeps):
        limiter = make_limiter()
        limiter.handle_rate_limit_response({"Retry-After": "1"})

        with pytest.raises(ConnectionError, match="boom"):
            async with RateLimitContext(limiter, timeout=1.0):
                raise ConnectionError("boom")

        assert limiter.state.consecutive_failures == 1
        assert limiter.state.current_backoff == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_timeout_raises_with_operation_name(self, make_limiter, sleeps):
        limiter = make_limiter(burst_capacity=1)
        assert await limiter.acquire()

        ctx = RateLimitContext(limiter, timeout=0.1, operation="GET_groups")
        with pytest.raises(TimeoutError, match="Rate limit timeout for operation: GET_groups"):
            async with ctx:
                pytest.fail("body must not run when no token is granted")

        assert ctx.acquired is False


class TestCreateRateLimiter:
    def test_defaults(self, clock):
        limiter = create_rate_limiter()

        assert isinstance(limiter, EnhancedTokenBucketRateLimiter)
        assert limiter.config.requests_per_minute == 300
        assert limiter.config.burst_capacity == 1
        assert limiter.config.enable_rate_limiting is True
        assert limiter.config.enable_queuing is True
        assert limiter.config.max_queue_size == 1000
        assert limiter.config.adaptive_rate_limiting is True
        assert limiter.state.tokens == 1.0

    def test_custom_arguments_are_applied(self, clock):
        limiter = create_rate_limiter(
            requests_per_minute=120,
            burst_capacity=10,
            enable_rate_limiting=False,
            enable_queuing=False,
            max_queue_size=5,
            adaptive_rate_limiting=False,
        )

        assert limiter.config.requests_per_minute == 120
        assert limiter.config.burst_capacity == 10
        assert limiter.config.enable_rate_limiting is False
        assert limiter.config.enable_queuing is False
        assert limiter.config.max_queue_size == 5
        assert limiter.config.adaptive_rate_limiting is False
        assert limiter.state.tokens == 10.0

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError, match="burst_capacity cannot exceed requests_per_minute"):
            create_rate_limiter(requests_per_minute=5, burst_capacity=6)

    def test_token_bucket_alias(self):
        assert TokenBucketRateLimiter is EnhancedTokenBucketRateLimiter
