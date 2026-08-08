import asyncio
from aws_lambda_powertools import Logger
from aiohttp import ClientSession

logger = Logger()


async def get_http_session() -> ClientSession:
    #return await asyncio.to_thread(get_http_session_sync)
    http_session = get_http_session_sync()
    logger.info("HTTP Session loaded. Just checking if async worked")
    return http_session

def get_http_session_sync() -> ClientSession:
    return ClientSession()
