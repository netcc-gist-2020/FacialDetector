import asyncio

results = []

async def async_generator(t):
	await asyncio.sleep(t)
	results.append(t)
	return t

async def monitor():
	while True:
		print("asdf")
		if len(results) > 0:
			return True
		await asyncio.sleep(0.1)

async def main():
	print(0)

	task1 = asyncio.ensure_future(async_generator(5))
	task2 = asyncio.ensure_future(async_generator(2))
	monitor_task = asyncio.ensure_future(monitor())

	#await asyncio.gather(task1, task2)
	await monitor_task

	print(results)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())