import asyncio
import websockets
import facialdetector
import json
import cv2

async def accept(websocket, path):
    cap, facial_detector = facialdetector.start()
    current_exp = None

    while True:
        result = facialdetector.run(cap, facial_detector)
        #print(result)
        #frame = facial_detector.frame

        frame = facial_detector.frame

        if result != None:
            if current_exp != result:
                current_exp = result

                try:
                    msg = {'absence': result[0], 'exp': result[1], 'eye_dir': result[2]}

                    print(msg)

                    await websocket.send(json.dumps(msg));
                    
                except TypeError:
                    pass




if __name__ == "__main__":
    websoc_svr = websockets.serve(accept, "localhost", 3000)
    
    asyncio.get_event_loop().run_until_complete(websoc_svr);
    asyncio.get_event_loop().run_forever();

    print("started")