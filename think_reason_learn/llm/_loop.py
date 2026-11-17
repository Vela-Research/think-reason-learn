"""One event loop, owned by the library, on which every model call runs.

API clients, limiters and caches are bound to the loop that creates them. Running
all calls on one long-lived loop in a background thread lets sync code (scripts,
``fit``) and async code (notebooks, apps, ``afit``) share them safely, with no
``nest_asyncio``.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, wait
from contextvars import ContextVar
from typing import Final

_lock: Final = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_fork_resets: Final[list[Callable[[], None]]] = []
_stop: Final[ContextVar[threading.Event | None]] = ContextVar(
    "think_reason_learn_stop", default=None
)
"""Set by the caller, from its own thread, when it cancels the run."""


def io_loop() -> asyncio.AbstractEventLoop:
    """The library's event loop, started on first use."""
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever, name="think-reason-learn-io", daemon=True
            ).start()
        return _loop


def run_sync[T](coro: Coroutine[object, object, T]) -> T:
    """Run ``coro`` on the library's loop and wait for it.

    Works from scripts and from inside a running event loop (e.g. a notebook
    cell). Ctrl-C cancels the work: no call starts after it, and it returns
    once the work has stopped (a second Ctrl-C stops waiting).

    Raises:
        RuntimeError: Called from the library's own loop.
    """
    loop = io_loop()
    if _on(loop):
        coro.close()
        raise RuntimeError("Use the async API (e.g. afit) inside the library's loop.")
    run = _Run(coro, loop)
    try:
        return _wait(run.future)
    except KeyboardInterrupt:
        run.stop()
        _wait_done(run.future)
        raise


async def run_async[T](coro: Coroutine[object, object, T]) -> T:
    """Await ``coro`` on the library's loop from any event loop.

    Cancelling the caller cancels the work: no call starts after the cancel,
    and the caller's ``CancelledError`` comes once the work has stopped, its
    calls recorded (cancelling again stops waiting).
    """
    loop = io_loop()
    if _on(loop):
        return await coro
    run = _Run(coro, loop)
    done = asyncio.wrap_future(run.future)
    try:
        return await asyncio.shield(done)
    except asyncio.CancelledError:
        run.stop()
        await asyncio.wait([done])
        if not done.cancelled():
            done.exception()  # retrieved: the caller cancelled, so it sees that
        raise


def stop_requested() -> bool:
    """Whether the caller has cancelled the run this code is part of."""
    stop = _stop.get()
    return stop is not None and stop.is_set()


def on_fork(reset: Callable[[], None]) -> None:
    """Run ``reset`` in a forked child, to drop state bound to the parent's loop."""
    _fork_resets.append(reset)


def _on(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _wait[T](future: Future[T]) -> T:
    _wait_done(future)
    return future.result()


def _wait_done[T](future: Future[T]) -> None:
    # Poll, so Ctrl-C reaches this thread on every platform.
    while not future.done():
        wait([future], timeout=0.5)


class _Run[T]:
    """``coro`` running on the library's loop, which the caller can stop.

    The cancel crosses threads, so the loop may start calls before it takes
    effect; ``stop`` therefore sets a flag first, from the caller's thread,
    which every call checks before it is sent (``stop_requested``).
    """

    def __init__(
        self, coro: Coroutine[object, object, T], loop: asyncio.AbstractEventLoop
    ) -> None:
        """Start ``coro`` on ``loop``."""
        self._loop = loop
        self._stop = threading.Event()
        self._task: asyncio.Task[object] | None = None
        self.future = asyncio.run_coroutine_threadsafe(self._run(coro), loop)

    def stop(self) -> None:
        """From the caller's thread: start no call from now on, and cancel."""
        self._stop.set()
        self._loop.call_soon_threadsafe(self._cancel)

    async def _run(self, coro: Coroutine[object, object, T]) -> T:
        self._task = asyncio.current_task()
        if self._stop.is_set():  # stopped before it started
            coro.close()
            raise asyncio.CancelledError
        _stop.set(self._stop)  # seen by the tasks this one starts, too
        return await coro

    def _cancel(self) -> None:
        # On the loop, like _run: a task not started yet stops itself there.
        if self._task is not None:
            self._task.cancel()


def _forget_loop() -> None:
    """A forked child has no loop thread: start a new loop on first use."""
    global _loop
    _loop = None
    for reset in _fork_resets:
        reset()


os.register_at_fork(after_in_child=_forget_loop)
