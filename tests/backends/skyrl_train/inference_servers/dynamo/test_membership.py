"""Concurrent requests share membership checks without releasing unchecked results."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from skyrl.backends.skyrl_train.inference_servers.dynamo.membership import (
    CohortInvalidated,
    MembershipGate,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_cohort_checks_once_before_and_after_all_requests(changed):
    check = AsyncMock(side_effect=[None, RuntimeError("changed")] if changed else None)
    gate = MembershipGate(check, Mock())
    started = 0
    all_started, release_last = asyncio.Event(), asyncio.Event()

    async def request(index):
        nonlocal started
        async with gate.attempt():
            started += 1
            if started == 16:
                all_started.set()
            await all_started.wait()
            if index == 15:
                await release_last.wait()
        return index

    tasks = [asyncio.create_task(request(i)) for i in range(16)]
    await all_started.wait()
    await asyncio.sleep(0)
    assert check.await_count == 1
    assert not any(task.done() for task in tasks)
    release_last.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert check.await_count == 2
    if changed:
        assert all(isinstance(result, RuntimeError) for result in results)
    else:
        assert results == list(range(16))
    assert not gate._phases
    assert not gate._current


@pytest.mark.asyncio
async def test_cancellation_discards_peers_and_drains_cohort():
    check = AsyncMock()
    fence = Mock()
    gate = MembershipGate(check, fence)
    entered, release = asyncio.Event(), asyncio.Event()

    async def request():
        async with gate.attempt():
            entered.set()
            await release.wait()

    first = asyncio.create_task(request())
    await entered.wait()
    second = asyncio.create_task(request())
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert isinstance(results[1], CohortInvalidated)
    assert isinstance(results[1].__cause__, asyncio.CancelledError)
    fence.assert_called_once_with()
    assert check.await_count == 2


@pytest.mark.asyncio
async def test_precheck_failure_never_runs_requests_and_next_phase_is_fresh():
    check = AsyncMock(side_effect=[RuntimeError("before"), None, None])
    gate = MembershipGate(check, Mock())
    with pytest.raises(CohortInvalidated) as error:
        async with gate.attempt():
            pytest.fail("request started before membership was validated")
    assert str(error.value.__cause__) == "before"
    async with gate.attempt():
        pass
    assert check.await_count == 3


@pytest.mark.asyncio
async def test_cancel_while_draining_preserves_caller_identity_and_holds_seal_task():
    checking_after, release_check = asyncio.Event(), asyncio.Event()
    fence = Mock()
    count = 0

    async def check():
        nonlocal count
        count += 1
        if count == 2:
            checking_after.set()
            await release_check.wait()

    gate = MembershipGate(check, fence)

    async def request():
        async with gate.attempt():
            await asyncio.sleep(0)

    tasks = [asyncio.create_task(request()) for _ in range(2)]
    await checking_after.wait()
    phase = next(iter(gate._phases.values()))
    assert isinstance(phase.seal_task, asyncio.Task)
    tasks[0].cancel()
    await asyncio.sleep(0)
    fence.assert_called_once_with()
    assert not any(task.done() for task in tasks)
    release_check.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert isinstance(results[1], CohortInvalidated)
    assert tasks[0].cancelled() and not tasks[1].cancelled()
    assert not gate._phases and not gate._current


@pytest.mark.asyncio
async def test_peers_get_distinct_exceptions_chained_to_one_failure():
    checking, release = asyncio.Event(), asyncio.Event()

    async def check():
        checking.set()
        await release.wait()
        raise RuntimeError("membership changed")

    gate = MembershipGate(check, Mock())

    async def request():
        async with gate.attempt():
            pytest.fail("request ran after failed pre-check")

    tasks = [asyncio.create_task(request()) for _ in range(4)]
    await checking.wait()
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, CohortInvalidated) for result in results)
    assert len({id(result) for result in results}) == 4
    assert len({id(result.__cause__) for result in results}) == 1


@pytest.mark.asyncio
async def test_sealing_previous_phase_does_not_forget_new_active_cohort():
    previous_sealing, release_previous = asyncio.Event(), asyncio.Event()
    current_started, release_current = asyncio.Event(), asyncio.Event()
    checks = 0

    async def check():
        nonlocal checks
        checks += 1
        if checks == 2:
            previous_sealing.set()
            await release_previous.wait()

    gate = MembershipGate(check, Mock())

    async def request(wait):
        async with gate.attempt():
            if wait:
                current_started.set()
                await release_current.wait()

    first = asyncio.create_task(request(False))
    await previous_sealing.wait()
    second = asyncio.create_task(request(True))
    await current_started.wait()
    release_previous.set()
    await first
    third = asyncio.create_task(request(True))
    await asyncio.sleep(0)
    assert checks == 3
    release_current.set()
    await asyncio.gather(second, third)
    assert checks == 4
    assert not gate._phases and not gate._current
