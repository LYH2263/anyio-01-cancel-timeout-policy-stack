from __future__ import annotations

import threading
from typing import Any

import pytest

import anyio
import anyio.lowlevel
from anyio import (
    CancelScope,
    CapacityLimiter,
    DeadlineMerge,
    OffloadCoupling,
    TimeoutPolicyFrame,
    TimeoutSurface,
    current_effective_deadline,
    current_time,
    current_timeout_policy,
    fail_after,
    fail_at,
    get_effective_timeout_policy,
    move_on_after,
    move_on_at,
    push_timeout_policy,
    sleep,
    to_thread,
)
from anyio._core import _policy
from anyio.from_thread import start_blocking_portal

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings(
        "ignore:The TerminalReporter.writer attribute is deprecated"
        ":pytest.PytestDeprecationWarning:"
    ),
]

# --------------------------------------------------------------------------------
# Exit 0: empty stack is exactly the baseline AnyIO behavior
# --------------------------------------------------------------------------------


async def test_empty_stack_views_are_none() -> None:
    assert current_timeout_policy() is None
    assert get_effective_timeout_policy() is None


async def test_empty_stack_fail_after_still_raises() -> None:
    with pytest.raises(TimeoutError):
        with fail_after(0.1):
            await sleep(1)


async def test_empty_stack_fail_at_still_raises() -> None:
    with pytest.raises(TimeoutError):
        with fail_at(current_time() + 0.1):
            await sleep(1)


async def test_empty_stack_move_on_swallows() -> None:
    with move_on_after(0.1) as scope:
        await sleep(1)

    assert scope.cancelled_caught


async def test_empty_stack_effective_deadline_is_infinite() -> None:
    assert current_effective_deadline() == float("inf")


async def test_move_on_timer_starts_at_factory_call() -> None:
    # v4 semantic: the move_on deadline is fixed when the object is created,
    # not when the with block is entered
    scope = move_on_after(0.1)
    await sleep(0.2)
    with scope:
        await anyio.lowlevel.checkpoint()

    assert scope.cancelled_caught


async def test_fail_timer_starts_on_enter() -> None:
    # v4 semantic: fail_after fixes the deadline when entering the with block,
    # so creating it early does not consume any of the budget
    cm = fail_after(0.3)
    await sleep(0.2)
    with cm:
        await sleep(0.05)


# --------------------------------------------------------------------------------
# Exit 1: frame timeout surfaces (fail / move_on / scope_only)
# --------------------------------------------------------------------------------


async def test_fail_frame_raises_timeout() -> None:
    with pytest.raises(TimeoutError, match="^frame deadline$"):
        with push_timeout_policy(delay=0.1, reason="frame deadline"):
            assert get_effective_timeout_policy() is not None
            await sleep(1)


async def test_fail_frame_is_the_default_surface() -> None:
    with pytest.raises(TimeoutError):
        with push_timeout_policy(delay=0.1):
            await sleep(1)


async def test_move_on_frame_swallows_deadline() -> None:
    with push_timeout_policy(
        delay=0.1, timeout_surface=TimeoutSurface.move_on
    ) as frame:
        await sleep(1)

    assert frame.cancelled_caught


async def test_scope_only_frame_records_cancelled_caught() -> None:
    with push_timeout_policy(
        delay=0.1, timeout_surface=TimeoutSurface.scope_only
    ) as frame:
        await sleep(1)

    assert frame.cancelled_caught
    assert current_timeout_policy() is None


async def test_frame_continues_execution_for_non_fail_surfaces() -> None:
    ran = False
    with push_timeout_policy(delay=0.1, timeout_surface="move_on"):
        await sleep(1)

    # execution resumes right after the frame boundary
    ran = True
    assert ran


async def test_frame_popped_on_arbitrary_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        with push_timeout_policy(delay=10):
            raise ValueError("boom")

    assert current_timeout_policy() is None


async def test_inner_pop_restores_outer_user_frame() -> None:
    with push_timeout_policy(delay=10, timeout_surface=TimeoutSurface.move_on) as outer:
        with push_timeout_policy(delay=1, timeout_surface="fail") as inner:
            assert current_timeout_policy() is inner
            assert get_effective_timeout_policy().frame is inner

        assert current_timeout_policy() is outer

    assert current_timeout_policy() is None


# --------------------------------------------------------------------------------
# Exit 2: the four timeout helpers fork by timeout_surface
# --------------------------------------------------------------------------------


async def test_move_on_after_under_fail_frame_raises_at_boundary() -> None:
    with pytest.raises(TimeoutError):
        with push_timeout_policy(delay=0.1):
            with move_on_after(10):
                await sleep(1)


async def test_move_on_at_under_fail_frame_raises_at_boundary() -> None:
    with pytest.raises(TimeoutError):
        with push_timeout_policy(delay=0.1):
            with move_on_at(current_time() + 10):
                await sleep(1)


async def test_fail_after_under_move_on_frame_moves_on() -> None:
    with push_timeout_policy(
        delay=0.1, timeout_surface=TimeoutSurface.move_on
    ) as frame:
        with fail_after(10):
            await sleep(1)

    assert frame.cancelled_caught


async def test_fail_at_under_scope_only_frame_does_not_raise() -> None:
    with push_timeout_policy(
        delay=0.1, timeout_surface=TimeoutSurface.scope_only
    ) as frame:
        with fail_at(current_time() + 10):
            await sleep(1)

    assert frame.cancelled_caught


async def test_inner_fail_helpers_own_timeout_still_raises() -> None:
    with pytest.raises(TimeoutError):
        with push_timeout_policy(delay=10, timeout_surface=TimeoutSurface.move_on):
            with fail_after(0.1):
                await sleep(1)


async def test_inner_move_on_helpers_own_deadline_still_moves_on() -> None:
    with push_timeout_policy(delay=10, timeout_surface=TimeoutSurface.fail):
        with move_on_after(0.1) as scope:
            await sleep(1)

        assert scope.cancelled_caught
        # The frame deadline was not reached, so no TimeoutError escapes
        await anyio.lowlevel.checkpoint()


async def test_helper_timing_points_preserved_under_frame() -> None:
    with push_timeout_policy(delay=10, timeout_surface=TimeoutSurface.fail):
        # move_on_after budget still starts at the factory call
        scope = move_on_after(0.1)
        await sleep(0.2)
        with scope:
            await anyio.lowlevel.checkpoint()

        assert scope.cancelled_caught

        # fail_after budget still starts on enter
        cm = fail_after(0.3)
        await sleep(0.2)
        with cm:
            await sleep(0.05)


# --------------------------------------------------------------------------------
# Exit 3: deadline_merge modes
# --------------------------------------------------------------------------------


async def test_strict_min_tightens_with_outer_frame() -> None:
    with push_timeout_policy(delay=0.3, timeout_surface=TimeoutSurface.scope_only):
        outer_deadline = get_effective_timeout_policy().deadline
        with push_timeout_policy(
            delay=10,
            timeout_surface=TimeoutSurface.scope_only,
            deadline_merge=DeadlineMerge.strict_min,
        ) as inner:
            assert inner.resolved_deadline == outer_deadline
            assert get_effective_timeout_policy().deadline == outer_deadline


async def test_active_frame_reports_top_frame_only() -> None:
    with push_timeout_policy(delay=0.3, timeout_surface=TimeoutSurface.scope_only):
        with push_timeout_policy(
            delay=10,
            timeout_surface=TimeoutSurface.scope_only,
            deadline_merge=DeadlineMerge.active_frame,
        ) as inner:
            policy = get_effective_timeout_policy()
            assert policy is not None
            assert policy.deadline == inner.resolved_deadline
            # physical enclosing scopes still tighten regardless
            assert current_effective_deadline() < inner.resolved_deadline


async def test_inherit_only_allows_tightening() -> None:
    with push_timeout_policy(delay=0.5, timeout_surface=TimeoutSurface.scope_only):
        with push_timeout_policy(
            delay=0.1,
            timeout_surface=TimeoutSurface.scope_only,
            deadline_merge=DeadlineMerge.inherit_only,
        ) as inner:
            assert get_effective_timeout_policy().deadline == inner.resolved_deadline


async def test_inherit_only_rejects_loosening() -> None:
    with push_timeout_policy(delay=0.2, timeout_surface=TimeoutSurface.scope_only):
        with pytest.raises(ValueError, match="only tighten"):
            with push_timeout_policy(
                delay=10,
                timeout_surface=TimeoutSurface.scope_only,
                deadline_merge=DeadlineMerge.inherit_only,
            ):
                pass

        assert current_timeout_policy() is not None


async def test_frame_tightens_effective_deadline_for_arbitrary_awaits() -> None:
    with push_timeout_policy(delay=0.3, timeout_surface=TimeoutSurface.scope_only):
        assert current_effective_deadline() < float("inf")


# --------------------------------------------------------------------------------
# Exit 4: CancelScope exit behavior visible while a frame is active
# --------------------------------------------------------------------------------


async def test_manual_cancel_scope_exit_under_frame() -> None:
    with push_timeout_policy(delay=10, timeout_surface=TimeoutSurface.scope_only):
        scope = CancelScope()
        with scope:
            scope.cancel()
            await anyio.lowlevel.checkpoint()

        assert scope.cancelled_caught

    assert current_timeout_policy() is None


async def test_shielded_frame_not_cancelled_by_outer_scope() -> None:
    completed = False

    async def canceller(scope: CancelScope) -> None:
        await sleep(0.1)
        scope.cancel()

    outer = CancelScope()
    async with anyio.create_task_group() as tg:
        with outer:
            tg.start_soon(canceller, outer)
            with push_timeout_policy(
                delay=10,
                timeout_surface=TimeoutSurface.scope_only,
                shield=True,
            ):
                # the outer cancellation at 0.1s must not interrupt this wait
                await sleep(0.3)

            completed = True
            await anyio.lowlevel.checkpoint()

    assert completed


# --------------------------------------------------------------------------------
# Exit 5: to_thread offload coupling (runs on every backend, incl. trio)
# --------------------------------------------------------------------------------


async def test_thread_follow_host_abandon() -> None:
    release = threading.Event()
    finished = threading.Event()

    def worker() -> str:
        release.wait(5)
        finished.set()
        return "done"

    import time

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        with push_timeout_policy(
            delay=0.2,
            offload_coupling=OffloadCoupling.thread_follow_host,
            abandon_on_cancel=True,
        ):
            await to_thread.run_sync(worker)

    assert time.perf_counter() - start < 1.0
    assert not finished.is_set()
    release.set()
    assert finished.wait(2)


async def test_thread_follow_host_waits_when_not_abandoned() -> None:
    # The frame deadline expires while the (non-abandoned) worker thread is
    # running. The worker wait is shielded and attached to the parent scope, so
    # the host runs the thread to completion instead of bailing out early (same
    # observable behavior as a plain to_thread call under fail_after).
    def worker() -> str:
        import time

        time.sleep(0.3)
        return "done"

    import time

    start = time.perf_counter()
    with push_timeout_policy(
        delay=0.1,
        timeout_surface=TimeoutSurface.scope_only,
        offload_coupling=OffloadCoupling.thread_follow_host,
        abandon_on_cancel=False,
    ):
        result = await to_thread.run_sync(worker)

    assert result == "done"
    assert time.perf_counter() - start >= 0.25


async def test_thread_follow_host_normal_completion() -> None:
    def worker(x: int) -> int:
        return x + 1

    with push_timeout_policy(
        delay=5,
        offload_coupling=OffloadCoupling.thread_follow_host,
        abandon_on_cancel=False,
    ):
        assert await to_thread.run_sync(worker, 1) == 2


async def test_limiter_frame_selects_budget() -> None:
    limiter = CapacityLimiter(1)
    seen = {}

    def worker() -> None:
        seen["borrowed"] = limiter.borrowed_tokens

    with push_timeout_policy(
        delay=5,
        offload_coupling=OffloadCoupling.limiter_frame,
        limiter=limiter,
    ):
        await to_thread.run_sync(worker)

    assert seen == {"borrowed": 1}
    assert limiter.borrowed_tokens == 0


async def test_explicit_limiter_overrides_limiter_frame() -> None:
    frame_limiter = CapacityLimiter(1)
    caller_limiter = CapacityLimiter(2)
    seen = {}

    def worker() -> None:
        seen["borrowed_frame"] = frame_limiter.borrowed_tokens
        seen["borrowed_caller"] = caller_limiter.borrowed_tokens

    with push_timeout_policy(
        delay=5,
        offload_coupling=OffloadCoupling.limiter_frame,
        limiter=frame_limiter,
    ):
        await to_thread.run_sync(worker, limiter=caller_limiter)

    assert seen == {"borrowed_frame": 0, "borrowed_caller": 1}


async def test_offload_coupling_does_nothing_without_flags() -> None:
    def worker() -> int:
        return 42

    with push_timeout_policy(
        delay=5, abandon_on_cancel=True, limiter=CapacityLimiter(1)
    ):
        # default limiter and default abandon behavior apply
        assert await to_thread.run_sync(worker) == 42


# --------------------------------------------------------------------------------
# Exit 6: task tree binding — child inheritance, sibling isolation
# --------------------------------------------------------------------------------


async def test_child_task_inherits_frame_but_pushes_are_local() -> None:
    seen = []

    async def child() -> None:
        seen.append(current_timeout_policy())

    with push_timeout_policy(delay=5, timeout_surface=TimeoutSurface.move_on) as frame:
        async with anyio.create_task_group() as tg:
            tg.start_soon(child)

    assert seen == [frame]


async def test_concurrent_host_tasks_do_not_share_stacks() -> None:
    observed = {}

    async def host(name: str, surface: Any) -> None:
        observed[name] = None
        try:
            with push_timeout_policy(delay=0.1, timeout_surface=surface):
                observed[name] = current_timeout_policy()
                await sleep(1)
        except TimeoutError:
            observed[f"{name}_raised"] = True

    async with anyio.create_task_group() as tg:
        tg.start_soon(host, "a", TimeoutSurface.fail)
        tg.start_soon(host, "b", TimeoutSurface.move_on)

    assert observed["a"].timeout_surface is TimeoutSurface.fail
    assert observed["b"].timeout_surface is TimeoutSurface.move_on
    assert observed.get("a_raised") is True
    assert "b_raised" not in observed


async def test_frame_deadline_propagates_to_spawned_children() -> None:
    async def child() -> None:
        await sleep(10)

    with pytest.raises(TimeoutError):
        with push_timeout_policy(delay=0.1):
            async with anyio.create_task_group() as tg:
                tg.start_soon(child)


# --------------------------------------------------------------------------------
# Exit 7: blocking portal forks the stack
# --------------------------------------------------------------------------------


async def test_portal_spawned_task_does_not_inherit_frame(
    deactivate_blockbuster: None,
) -> None:
    done = threading.Event()
    observed = {}

    async def portal_task() -> None:
        observed["top"] = current_timeout_policy()
        observed["effective"] = get_effective_timeout_policy()
        done.set()

    def spawn(portal: Any) -> None:
        portal.start_task_soon(portal_task)
        done.wait(2)

    with push_timeout_policy(delay=5, timeout_surface=TimeoutSurface.scope_only):
        with start_blocking_portal() as portal:
            await to_thread.run_sync(spawn, portal)
            assert current_timeout_policy() is not None

    assert observed["top"] is None
    assert observed["effective"] is None


# --------------------------------------------------------------------------------
# Exit 8: library-internal scopes do not subscribe to the stack
# --------------------------------------------------------------------------------


async def test_detached_policy_hides_the_stack() -> None:
    with push_timeout_policy(delay=5):
        assert current_timeout_policy() is not None
        with _policy.detached_timeout_policy():
            assert current_timeout_policy() is None
            assert get_effective_timeout_policy() is None

        assert current_timeout_policy() is not None


async def test_connect_tcp_refused_under_tight_frame() -> None:
    # the library-internal happy eyeballs scope must not subscribe to the
    # frame; an immediate refusal still surfaces as the baseline error
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        with push_timeout_policy(delay=2, timeout_surface=TimeoutSurface.scope_only):
            with pytest.raises(ConnectionRefusedError):
                await anyio.connect_tcp("127.0.0.1", port)
    finally:
        sock.close()


# --------------------------------------------------------------------------------
# Exit 9: anyio.run subscribes; the pytest TestRunner does not retain stacks
# --------------------------------------------------------------------------------


def test_run_subscribes_timeout_policy() -> None:
    # anyio.run is an entry point that can subscribe to the policy stack for
    # its whole root task
    async def main() -> None:
        policy = get_effective_timeout_policy()
        assert policy is not None
        await sleep(1)

    for backend in ("asyncio", "trio"):
        spec = TimeoutPolicyFrame(delay=0.1, timeout_surface=TimeoutSurface.fail)
        with pytest.raises(TimeoutError):
            anyio.run(main, backend=backend, timeout_policy=spec)

        spec = TimeoutPolicyFrame(delay=5, timeout_surface=TimeoutSurface.scope_only)
        anyio.run(main, backend=backend, timeout_policy=spec)


def _runner_isolation_conftest(backend_name: str) -> str:
    return f"""
import pytest


@pytest.fixture(scope="module")
def anyio_backend():
    return {backend_name!r}
"""


_RUNNER_ISOLATION_TESTS = """
import pytest
import anyio._core._policy as policy
from anyio import current_timeout_policy, push_timeout_policy, get_effective_timeout_policy

pytestmark = pytest.mark.anyio


# A module scoped async fixture keeps the same TestRunner task alive across the
# three tests below (otherwise the runner lease is dropped between items).
@pytest.fixture(scope="module")
async def keep_runner():
    yield


async def test_a_leaks_a_raw_frame(keep_runner) -> None:
    # simulate a forgotten pop by writing the mount point directly
    policy._policy_stack.set(("leaked",))
    # deliberately do NOT reset the token


async def test_b_sees_an_empty_stack(keep_runner) -> None:
    assert current_timeout_policy() is None
    assert get_effective_timeout_policy() is None


async def test_c_can_still_push_its_own_frames(keep_runner) -> None:
    with push_timeout_policy(delay=5):
        assert current_timeout_policy() is not None
    assert current_timeout_policy() is None
"""


@pytest.mark.parametrize("backend_name", ["asyncio", "trio"])
def test_test_runner_does_not_retain_business_stack(
    pytester: pytest.Pytester, backend_name: str
) -> None:
    pytester.makeconftest(_runner_isolation_conftest(backend_name))
    pytester.makepyfile(_RUNNER_ISOLATION_TESTS)
    result = pytester.runpytest(
        "-v", "-p", "anyio", "-p", "no:asyncio", "-p", "no:trio"
    )
    result.assert_outcomes(passed=3)
