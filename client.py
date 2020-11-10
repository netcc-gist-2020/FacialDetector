import asyncio
import websockets
import json

socket_url = "116.89.189.56:8080"

async def connect():
    async with websockets.connect(f"ws://{socket_url}") as websocket:
        opening = {
            "type":"open",
            "data": {}
        }
        await websocket.send(json.dumps(opening))
        data = await websocket.recv()
        print(data)
