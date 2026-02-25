import asyncio

from processor import processor


asyncio.set_event_loop(asyncio.new_event_loop())


async def async_handler(event, context):
    batch_failures = await processor.process_images(event['Records'])

    return batch_failures.model_dump()

def lambda_handler(event, context):
    loop = asyncio.get_event_loop()

    return loop.run_until_complete(async_handler(event, context))

if __name__ == "__main__":
    pass;
