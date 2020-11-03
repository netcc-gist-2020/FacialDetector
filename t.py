import asyncio

results = []

async def async_generator(t):
	while True:
		await asyncio.sleep(t)
		print(t)
		results.append(t)

		return "Ok"
		
	#return t

async def monitor():
	while True:
		
		if len(results) > 0:
			print(results)
			return True

		await asyncio.sleep(0.1)

async def main():


	task1 = asyncio.ensure_future(async_generator(5))
	task2 = asyncio.ensure_future(async_generator(2))
	monitor_task = asyncio.ensure_future(monitor())

	#await asyncio.gather(task1, task2)
	while True:
		#await monitor_task
		#results.clear()
		#monitor_task = asyncio.ensure_future(monitor())
		await task1

loop = asyncio.get_event_loop()
loop.run_until_complete(main())