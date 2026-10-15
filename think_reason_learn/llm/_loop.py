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
from collections.abc import Coroutine
from concurrent.futures import Future, wait
from typing import Final

_lock: Final = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


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
    cell). Ctrl-C cancels the work.

    Raises:
        RuntimeError: Called from the library's own loop.
    """
    loop = io_loop()
    if _on(loop):
        coro.close()
        raise RuntimeError("Use the async API (e.g. afit) inside the library's loop.")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return _wait(future)
    except KeyboardInterrupt:
        future.cancel()
        raise


async def run_async[T](coro: Coroutine[object, object, T]) -> T:
    """Await ``coro`` on the library's loop from any event loop."""
    loop = io_loop()
    if _on(loop):
        return await coro
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


def _on(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _wait[T](future: Future[T]) -> T:
    # Poll, so Ctrl-C reaches this thread on every platform.
    while not future.done():
        wait([future], timeout=0.5)
    return future.result()


def _forget_loop() -> None:
    """A forked child has no loop thread: start a new loop on first use."""
    global _loop
    _loop = None


os.register_at_fork(after_in_child=_forget_loop)
