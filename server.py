import logging
import time
log = logging.getLogger('api')
log.debug("Loading asyncio...")
start = time.time()

import asyncio

end = time.time()
log.debug(f"Loading asyncio took {end - start} seconds")

log.debug("Loading processor instance...")
start = time.time()

from processor import processor

end = time.time()
log.debug(f"Loading processor instance took {end - start} seconds")


asyncio.set_event_loop(asyncio.new_event_loop())


async def async_handler(event, context):
    batch_failures = await processor.process_images(event['Records'])

    return batch_failures.model_dump()

def lambda_handler(event, context):
    loop = asyncio.get_event_loop()

    return loop.run_until_complete(async_handler(event, context))

if __name__ == "__main__":
    pass;
