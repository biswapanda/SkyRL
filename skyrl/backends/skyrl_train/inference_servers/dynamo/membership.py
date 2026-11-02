"""Batch membership fencing: no result escapes before its cohort is checked."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace


class CohortInvalidated(RuntimeError):
    """A peer or membership check invalidated this request's serving cohort."""


@dataclass(frozen=True)
class _Phase:
    before: asyncio.Task[BaseException | None]
    done: asyncio.Future[BaseException | None]
    active: int = 0
    failure: BaseException | None = None
    seal_task: asyncio.Task[None] | None = None


class MembershipGate:
    """Share pre/post discovery across overlapping requests on each event loop.

    Failures and cancellation invalidate every result in the cohort. Completion
    is drained before an attempt exits, keeping the driver's update fence held.
    """

    def __init__(self, check: Callable[[], Awaitable[None]], fence: Callable[[], None]) -> None:
        self._check = check
        self._fence = fence
        self._current: dict[asyncio.AbstractEventLoop, object] = {}
        self._phases: dict[object, _Phase] = {}

    def _fail(self, key: object, failure: BaseException) -> None:
        phase = self._phases.get(key)
        if phase is not None and phase.failure is None:
            self._phases[key] = replace(phase, failure=failure)
            self._fence()

    async def _before(self, key: object) -> BaseException | None:
        try:
            await self._check()
        except BaseException as exc:
            self._fail(key, exc)
            return exc
        return None

    async def _seal(self, key: object) -> None:
        phase = self._phases[key]
        try:
            if await phase.before is None:
                await self._check()
        except BaseException as exc:
            self._fail(key, exc)
        finally:
            phase.done.set_result(self._phases.pop(key).failure)
            loop = asyncio.get_running_loop()
            if self._current.get(loop) is key:
                del self._current[loop]

    @asynccontextmanager
    async def attempt(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        key = self._current.get(loop)
        if key is None or not self._phases[key].active:
            key = object()
            self._current[loop] = key
            self._phases[key] = _Phase(asyncio.create_task(self._before(key)), loop.create_future())
        phase = self._phases[key]
        self._phases[key] = replace(phase, active=phase.active + 1)
        own_failure = None
        try:
            failure = await asyncio.shield(phase.before)
            if failure is not None:
                raise CohortInvalidated("Dynamo cohort membership pre-check failed") from failure
            yield
        except BaseException as exc:
            own_failure = exc
            self._fail(key, exc)
        finally:
            state = self._phases[key]
            self._phases[key] = replace(state, active=state.active - 1)
            if state.active == 1:
                self._phases[key] = replace(self._phases[key], seal_task=asyncio.create_task(self._seal(key)))
            while not phase.done.done():
                try:
                    await asyncio.shield(phase.done)
                except asyncio.CancelledError as exc:
                    own_failure = exc
                    self._fail(key, exc)
            if own_failure is not None:
                raise own_failure
            failure = phase.done.result()
            if failure is not None:
                raise CohortInvalidated("Dynamo cohort was invalidated before result admission") from failure
