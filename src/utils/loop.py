import asyncio
from asyncio import AbstractEventLoop
import threading


def ensure_loop() -> AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not loop.is_running():
        t = threading.Thread(target=loop.run_forever)
        t.start()

    return loop
