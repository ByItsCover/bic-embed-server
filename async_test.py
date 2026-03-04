import asyncio

async def async_fun(n = 10):
    for i in range(n):
        print("Fun Time", i)
        await asyncio.sleep(1)

async def async_fun2(n = 5):
    for i in range(n):
        print("Fun2 Time", i)
        await asyncio.sleep(2)

async def main():
    fun_task = asyncio.create_task(async_fun())
    fun2_task = asyncio.create_task(async_fun2())

    for i in range(5):
        print("Main Time", i)
        await asyncio.sleep(1)
    
    await fun_task
    await fun2_task

if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    loop.run_until_complete(main())
