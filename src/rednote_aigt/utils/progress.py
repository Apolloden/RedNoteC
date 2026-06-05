"""Progress helpers for long blocking library calls."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def log_heartbeat(
    logger: logging.Logger,
    message: str,
    interval_seconds: int = 30,
) -> Iterator[None]:
    """Log a heartbeat while a blocking call runs."""
    stop = threading.Event()
    start = time.monotonic()

    def worker() -> None:
        while not stop.wait(interval_seconds):
            elapsed = int(time.monotonic() - start)
            logger.info("%s still running after %ss", message, elapsed)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
