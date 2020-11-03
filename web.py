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
    #cap, facial_detector = facialdetector.start()
    onCamera = False
    isSpy = True
    
    user_name = await websocket.recv()
    user_name = user_name.strip()
    # print("receive: " + user_name)

    facial_detector = facialdetector.start()
    cap = facial_detector.cap

    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(facial_detector.run())

    asyncio.ensure_future(facial_detector.run())

    while True:
        result = facial_detector.info
        if not onCamera and result["absence"] == 'present':
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

        elif result["absence"] == 'absence':
            onCamera = False

        if facial_detector.updated == True:
            try:
                msg = {'type': 'exp', 'data': {'absence': result["absence"], 'expression': result["expression"], 'eye_dir': result["eye_dir"], 'sleepiness': result["sleepiness"], 'isSpy': isSpy}}

                print(msg)

                await websocket.send(json.dumps(msg))
                
            except TypeError:
                pass

        await asyncio.sleep(1)

if __name__ == "__main__":
    websoc_svr = websockets.serve(accept, "localhost", 3000)
    asyncio.get_event_loop().run_until_complete(websoc_svr)
    asyncio.get_event_loop().run_forever()