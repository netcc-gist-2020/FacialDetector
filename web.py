import asyncio
import websockets
import facialdetector
import numpy
import json
import cv2
import requests

async def spy_check(user_name, img):
    url = "http://116.89.189.53:8081/signin/face"
    params = {
        "meta-data":  {
            'name': user_name
        }
    }
    response = requests.post(url=url, files= {"user-face":img}, data=params)
    print(response)
    if response.status_code == 202:
        print("User Confimed!!")
        return False
    elif response.status_code == 401:
        print("You are Spy!!")
        return True
    else:
        print("Connection Error")
        return True
    

async def accept(websocket, path):
    cap, facial_detector = facialdetector.start()
    current_exp = None
    onCamera = False
    isSpy = True
    
    user_name = await websocket.recv()
    user_name = user_name.strip()
    # print("receive: " + user_name)
    while True:
        result = facialdetector.run(cap, facial_detector)

        if result != None:
            if not onCamera and result[0] == 'present':
                onCamera = True
                ret, frame = cap.read()

                if ret == False:
                    continue
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, imgencode = cv2.imencode('.jpg', frame, encode_param)
                stringImg = imgencode.tobytes()
                
                # spy_check doesn't work with late response
                isSpy = await spy_check(user_name, stringImg)

                try:
                    msg = {'type':'spy', 'data': {'img': "spy"}}
                    print(msg)
                    await websocket.send(json.dumps(msg))
                except TypeError:
                    print("type error")
                    pass

            elif result[0] == 'absence':
                onCamera = False

            if current_exp != result:
                current_exp = result

                try:
                    msg = {'type': 'exp', 'data': {'absence': result[0], 'expression': result[1], 'eye_dir': result[2], 'isSpy': isSpy}}

                    # print(msg)

                    await websocket.send(json.dumps(msg))
                    
                except TypeError:
                    pass

if __name__ == "__main__":
    websoc_svr = websockets.serve(accept, "localhost", 3000)
    asyncio.get_event_loop().run_until_complete(websoc_svr)
    asyncio.get_event_loop().run_forever()