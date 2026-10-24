from __future__ import annotations

__all__ = (
    "DeadlineMerge",
    "EffectiveTimeoutPolicy",
    "OffloadCoupling",
    "TimeoutPolicyFrame",
    "TimeoutSurface",
    "current_timeout_policy",
    "get_effective_timeout_policy",
    "push_timeout_policy",
)

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from enum import Enum, IntFlag
from typing import TYPE_CHECKING, Any

from ._eventloop import get_async_backend

if TYPE_CHECKING:
    from ._synchronization import CapacityLimiter
    from ._tasks import CancelScope


class DeadlineMerge(Enum):
    """
    How a pushed :class:`TimeoutPolicyFrame` combines its deadline with the
    deadlines in effect outside the frame.

    .. attribute:: strict_min

        The frame deadline is combined with the deadline of the enclosing
        cancel scope chain (and any frames below it) using the minimum (the
        tightest deadline always wins).

    .. attribute:: active_frame

        Only the stack top frame is consulted when reporting the policy view;
        enclosing *policy* frames are masked. Physical cancel scopes (including
        frames below) still cancel work as usual.

    .. attribute:: inherit_only

        The frame may only tighten the inherited effective deadline. Attempting
        to push a looser deadline raises :class:`ValueError`.
    """

    strict_min = "strict_min"
    active_frame = "active_frame"
    inherit_only = "inherit_only"


class TimeoutSurface(Enum):
    """
    How a timeout of a pushed frame is surfaced when the frame body is
    interrupted by the frame's own deadline.

    .. attribute:: fail

        Behave like :func:`~anyio.fail_at` / :func:`~anyio.fail_after`: once the
        frame deadline cancels the body, :class:`TimeoutError` is raised at the
        frame boundary.

    .. attribute:: move_on

        Behave like :func:`~anyio.move_on_at` / :func:`~anyio.move_on_after`:
        the frame deadline cancels the body but no exception is raised at the
        frame boundary.

    .. attribute:: scope_only

        The frame only participates in cancellation propagation (like a bare
        :class:`~anyio.CancelScope` with a deadline). It never converts the
        cancellation into a :class:`TimeoutError`; the outcome is observable
        through :attr:`TimeoutPolicyFrame.cancelled_caught`.
    """

    fail = "fail"
    move_on = "move_on"
    scope_only = "scope_only"


class OffloadCoupling(IntFlag):
    """
    Flags describing how work unloaded from the host task couples with the
    stack top frame.

    .. attribute:: thread_follow_host

        ``abandon_on_cancel`` of :func:`~anyio.to_thread.run_sync` follows the
        value set on the stack top frame.

    .. attribute:: limiter_frame

        :func:`~anyio.to_thread.run_sync` borrows its worker thread from the
        :class:`~anyio.CapacityLimiter` set on the stack top frame when the
        caller did not explicitly pass one.

    .. attribute:: portal_fork

        Tasks spawned from a thread via :class:`~anyio.from_thread.BlockingPortal`
        do not inherit the frames pushed by the host task.
    """

    thread_follow_host = 1
    limiter_frame = 2
    portal_fork = 4


_NO_COUPLING = OffloadCoupling(0)


@dataclass(frozen=True)
class TimeoutPolicyFrame:
    """
    A single frame of the task-tree timeout and cancellation policy stack.

    Create frames with :func:`push_timeout_policy` rather than instantiating
    this class directly.
    """

    deadline: float = math.inf
    delay: float | None = None
    deadline_merge: DeadlineMerge = DeadlineMerge.strict_min
    timeout_surface: TimeoutSurface | None = TimeoutSurface.fail
    offload_coupling: OffloadCoupling = _NO_COUPLING
    abandon_on_cancel: bool | None = None
    limiter: CapacityLimiter | None = None
    shield: bool = False
    reason: str | None = None
    _scope: CancelScope | None = field(default=None, compare=False, repr=False)
    _resolved_deadline: float = field(default=math.inf, compare=False, repr=False)

    def __post_init__(self) -> None:
        merge = self.deadline_merge
        if not isinstance(merge, DeadlineMerge):
            object.__setattr__(self, "deadline_merge", DeadlineMerge(merge))

        surface = self.timeout_surface
        if surface is not None and not isinstance(surface, TimeoutSurface):
            object.__setattr__(self, "timeout_surface", TimeoutSurface(surface))

        coupling = self.offload_coupling
        if not isinstance(coupling, OffloadCoupling):
            object.__setattr__(self, "offload_coupling", OffloadCoupling(coupling))

        if self.delay is not None and self.deadline != math.inf:
            raise ValueError('only one of "delay" and "deadline" may be provided')

    @property
    def resolved_deadline(self) -> float:
        """The absolute deadline the frame was entered with."""
        return self._resolved_deadline

    @property
    def cancelled_caught(self) -> bool:
        """
        ``True`` if the frame suppressed a cancellation that its own deadline
        triggered.
        """
        return self._scope is not None and self._scope.cancelled_caught


@dataclass(frozen=True)
class EffectiveTimeoutPolicy:
    """
    A read-only snapshot of the stack top merged with the current
    :class:`~anyio.CancelScope` tree.

    Returned by :func:`get_effective_timeout_policy`. ``None`` is returned when
    no frame has been pushed (or the stack is temporarily detached), in which
    case all baseline behavior is in effect.
    """

    frame: TimeoutPolicyFrame
    deadline: float

    @property
    def timeout_surface(self) -> TimeoutSurface | None:
        return self.frame.timeout_surface

    @property
    def offload_coupling(self) -> OffloadCoupling:
        return self.frame.offload_coupling

    @property
    def abandon_on_cancel(self) -> bool | None:
        return self.frame.abandon_on_cancel

    @property
    def limiter(self) -> CapacityLimiter | None:
        return self.frame.limiter


#: The policy stack. Each task holds its own copy of the value (via the standard
#: context variable propagation rules), so concurrent host tasks pushing frames
#: never see each other's frames, while tasks spawned within a frame inherit it.
_policy_stack: ContextVar[tuple[TimeoutPolicyFrame, ...]] = ContextVar(
    "anyio_timeout_policy_stack", default=()
)

#: Non-zero while library-internal cancel scopes are being entered that must
#: not subscribe to the application policy stack.
_policy_detached: ContextVar[int] = ContextVar(
    "anyio_timeout_policy_detached", default=0
)


def current_timeout_policy() -> TimeoutPolicyFrame | None:
    """
    Return the frame at the top of the timeout policy stack, or ``None`` if the
    stack is empty or detached.
    """
    if _policy_detached.get():
        return None

    stack = _policy_stack.get(())
    return stack[-1] if stack else None


def get_effective_timeout_policy() -> EffectiveTimeoutPolicy | None:
    """
    Return a merged view of the stack top frame and the deadlines currently in
    effect via the enclosing :class:`~anyio.CancelScope` tree (and the timeout
    helpers).

    :return: the effective policy, or ``None`` when no frame is in effect
    """
    frame = current_timeout_policy()
    if frame is None:
        return None

    physical_deadline = get_async_backend().current_effective_deadline()
    if frame.deadline_merge is DeadlineMerge.active_frame:
        deadline = frame.resolved_deadline
    else:
        deadline = min(frame.resolved_deadline, physical_deadline)

    return EffectiveTimeoutPolicy(frame=frame, deadline=deadline)


@contextmanager
def detached_timeout_policy() -> Iterator[None]:
    """
    Hide the policy stack for the duration of the block.

    Used by library-internal cancel scopes (such as the temporary shield scope
    used while waiting for task group children) so that they do not subscribe
    to the application policy stack.
    """
    token = _policy_detached.set(_policy_detached.get() + 1)
    try:
        yield
    finally:
        _policy_detached.reset(token)


@contextmanager
def isolated_timeout_policy_stack() -> Iterator[None]:
    """
    Run the block with an empty policy stack and restore the previous value on
    exit, even if the block leaked a push.

    Used by the pytest plugin's ``TestRunner`` entry point so that the runner
    machinery never subscribes to (or retains) a business task-tree stack.
    Frames pushed within the block still work normally.
    """
    token = _policy_stack.set(())
    try:
        yield
    finally:
        _policy_stack.reset(token)


@contextmanager
def fork_timeout_policy_stack() -> Iterator[None]:
    """
    Enter a block (a task spawned from outside the task tree, e.g. via a
    blocking portal) with an empty policy stack, without inheriting frames
    pushed by the host task.
    """
    token = _policy_stack.set(())
    try:
        yield
    finally:
        _policy_stack.reset(token)


def _resolve_deadline(frame: TimeoutPolicyFrame, current_time: float) -> float:
    requested = (
        current_time + frame.delay if frame.delay is not None else frame.deadline
    )
    inherited = get_async_backend().current_effective_deadline()
    if frame.deadline_merge is DeadlineMerge.active_frame:
        return requested
    elif frame.deadline_merge is DeadlineMerge.inherit_only:
        if inherited != math.inf and requested > inherited:
            raise ValueError(
                "an inherit_only timeout policy frame may only tighten the "
                "inherited effective deadline"
            )

        return requested

    return min(requested, inherited)


@contextmanager
def push_timeout_policy(
    *,
    deadline: float = math.inf,
    delay: float | None = None,
    deadline_merge: DeadlineMerge | str = DeadlineMerge.strict_min,
    timeout_surface: TimeoutSurface | str | None = TimeoutSurface.fail,
    offload_coupling: OffloadCoupling | int = _NO_COUPLING,
    abandon_on_cancel: bool | None = None,
    limiter: CapacityLimiter | None = None,
    shield: bool = False,
    reason: str | None = None,
) -> Iterator[TimeoutPolicyFrame]:
    """
    Push a timeout and cancellation policy frame onto the current task-tree
    stack.

    The frame is popped automatically when the context block is exited,
    including when the block exits via an exception. Popping restores the
    previous user frame; frames never fall back to a process-wide global
    value.

    :param deadline: absolute clock value when the frame deadline expires
    :param delay: relative time (in seconds) until the frame deadline expires;
        mutually exclusive with ``deadline``
    :param deadline_merge: how the frame deadline is combined with deadlines
        in effect outside the frame
    :param timeout_surface: how expiry of the frame deadline is surfaced
        (``fail`` raises :class:`TimeoutError`, ``move_on`` swallows the
        cancellation, ``scope_only`` only cancels and records
        ``cancelled_caught``)
    :param offload_coupling: flags selecting how work unloaded via
        :func:`~anyio.to_thread.run_sync` and blocking portals couples with
        this frame
    :param abandon_on_cancel: the ``abandon_on_cancel`` value enforced for
        :func:`~anyio.to_thread.run_sync` calls coupled via
        :attr:`OffloadCoupling.thread_follow_host`
    :param limiter: the capacity limiter selected for
        :func:`~anyio.to_thread.run_sync` calls coupled via
        :attr:`OffloadCoupling.limiter_frame`
    :param shield: shield the frame's cancel scope from external cancellation
    :param reason: message added to the raised :class:`TimeoutError` when
        ``timeout_surface`` is :attr:`TimeoutSurface.fail`
    """
    spec = TimeoutPolicyFrame(
        deadline=deadline,
        delay=delay,
        deadline_merge=deadline_merge,  # type: ignore[arg-type]
        timeout_surface=timeout_surface,  # type: ignore[arg-type]
        offload_coupling=offload_coupling,  # type: ignore[arg-type]
        abandon_on_cancel=abandon_on_cancel,
        limiter=limiter,
        shield=shield,
        reason=reason,
    )
    with enter_timeout_policy(spec) as frame:
        yield frame


@contextmanager
def enter_timeout_policy(spec: TimeoutPolicyFrame) -> Iterator[TimeoutPolicyFrame]:
    """Enter a pre-built policy frame spec (used by the ``anyio.run`` entry)."""
    backend = get_async_backend()
    resolved = _resolve_deadline(spec, backend.current_time())
    scope = backend.create_cancel_scope(deadline=resolved, shield=spec.shield)
    frame = replace(spec, _scope=scope, _resolved_deadline=resolved)

    stack_token: Token[Any] = _policy_stack.set(_policy_stack.get(()) + (frame,))
    try:
        with scope:
            yield frame

        if (
            frame.timeout_surface is TimeoutSurface.fail
            and scope.cancelled_caught
            and backend.current_time() >= scope.deadline
        ):
            if frame.reason:
                raise TimeoutError(frame.reason)
            else:
                raise TimeoutError
    finally:
        _policy_stack.reset(stack_token)
