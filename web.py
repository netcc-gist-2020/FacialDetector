import asyncio
import websockets
import facialdetector
import numpy
import json
import cv2

async def accept(websocket, path):
    cap, facial_detector = facialdetector.start()
    current_exp = None
    onCamera = False

    while True:
        result = facialdetector.run(cap, facial_detector)
        #print(result)
        #frame = facial_detector.frame
        frame = facial_detector.frame

        if result != None:
          
            if not onCamera and result[0] = 'present':
                onCamera = True
                ret, frame = cap.read()

                if ret == False:
                    continue

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, imgencode = cv2.imencode('.jpg', frame, encode_param)
                stringImg = numpy.array(imgencode).tobytes()

                msg = {'type':'spy', 'data': {'img': stringImg}}
                await websocket.send(json.jumps(msg))

            elif result[0] = 'absent':
                onCamera = False

            if current_exp != result:
                current_exp = result

                try:
                    msg = {'type': 'exp', 'data': {'absence': result[0], 'expression': result[1], 'eye_dir': result[2]}}

                    print(msg)

                    await websocket.send(json.dumps(msg))
                    
                except TypeError:
                    pass




if __name__ == "__main__":
    websoc_svr = websockets.serve(accept, "localhost", 3000)
    
    asyncio.get_event_loop().run_until_complete(websoc_svr)
    asyncio.get_event_loop().run_forever()

    print("started")