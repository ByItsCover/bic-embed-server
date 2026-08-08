import asyncio
from aiohttp import ClientSession


async def get_http_session() -> ClientSession:
    return await asyncio.to_thread(get_http_session_sync)

def get_http_session_sync() -> ClientSession:
    return ClientSession()
