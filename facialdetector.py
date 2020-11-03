import cv2
import numpy as np
import dlib
import math
import time
import json

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, model_from_json
from keras.preprocessing.image import img_to_array

import asyncio

def euc_dist(x1, y1, x2, y2):
    return math.sqrt((abs(x1-x2))**2+(abs(y1-y2))**2)

def landmarks_dist(n1, n2, landmarks):
    return euc_dist(landmarks.part(n1).x, landmarks.part(n1).y, landmarks.part(n2).x, landmarks.part(n2).y)




class FacialDetector:

    def __init__(self, detector, landmarker, exp_classifier, facerec, cap):
        self.detector = detector
        self.landmarker = landmarker
        self.exp_classifier = exp_classifier
        self.facerec = facerec 
        self.cap = cap

        self.gray = None
        self.frame = None

        # face descriptors from server
        self.target_face_descriptor = None

        # self.detector result
        self.target_face = None
        self.mean_eye_ratio = 0


        self.face_buffer = []
        self.info = {"absence": None, "expression": None, "eye_dir": None, "sleepiness": None}

        self.updated = True


        self.exp_labels = ('happy', 'neutral') #('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


    def set_frame(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    async def detect_timer(self, func, info_name, buffer_size, sleeptime):
        # notify on init, and when the result changes (after some threshold)
        try:
            result = func(self.target_face)
            buf = np.array([result])
            print(result)
        except:
            result = None
            buf = np.array([])

        while True:
            try:
                current_result = func(self.target_face)

                buf = np.append(buf, current_result)
                buf = buf[-buffer_size:]

                if len(buf) == buffer_size:
                    if np.sum(buf == current_result) > np.sum(buf == result):
                        buf = buf[buf!=result]
                        result = current_result

                        print(current_result)
                        self.updated = True

                    self.info[info_name] = result

            except ValueError:
                #print(info_name+ ": ValueError!")
                pass
            await asyncio.sleep(sleeptime)



    async def find_target(self, timer):

        while True:

            if self.target_face != None:
                # local search around previous target face
                face = self.target_face
                d = 300
                hrange = range(max(face.top()-d,0), min(face.bottom()+d,self.gray.shape[0]))
                wrange = range(max(face.left()-d,0), min(face.right()+d,self.gray.shape[1]))

                faces = self.detector(self.gray[hrange][:,wrange])

                if len(faces) == 0:
                    # go for global search
                    self.target_face = None
                    continue

                else:
                    new_face = faces[0]
                    self.target_face = dlib.rectangle(new_face.left()+wrange[0], new_face.top()+hrange[0], 
                                                    new_face.right()+wrange[0], new_face.bottom()+hrange[0])

            else:
                # global search
                #print("global search.")

                faces = self.detector(self.gray)
                tasks = []

                if len(faces) == 1:
                    face = faces[0]
                    self.target_face = face

                    try:
                        self.target_face_descriptor = await get_descriptor(face)
                    except:
                        pass
                    

                elif len(faces) > 0:

                    for face in faces:
                        tasks.append(asyncio.ensure_future(compare_face_target(face)))

                    task_monitor = asyncio.ensure_future(compare_face_target_monitor(0.33*timer, len(faces)))
                    self.target_face = await task_monitor

                    for task in tasks:
                        if not task.done():
                            task.cancel()
                else:
                    self.target_face = None

            await asyncio.sleep(timer)


    def detect_sleepy(self, face):
        try:
            y=self.compute_eye_ratio(face)
        except:
            raise ValueError("detect_sleepy: No face detected")

        return "sleepy" if y<1.4 else "awake"


    async def run(self):
        task_target = asyncio.ensure_future(self.find_target(0.1))


        task_absence = asyncio.ensure_future(self.detect_timer(lambda x: "absence" if x==None else "present", 
                                                                "absence", 120, 1))

        task_exp = asyncio.ensure_future(self.detect_timer(self.detect_expression, "expression", 20, 0.5))

        task_gazing = asyncio.ensure_future(self.detect_timer(self.detect_gazing, "eye_dir", 8, 0.5))

        task_sleepy = asyncio.ensure_future(self.detect_timer(self.detect_sleepy,"sleepiness", 60, 1))
        #task


        while True:
            frame = self.cap.read()[1]
            self.set_frame(frame)

            if self.target_face != None:
                cv2.rectangle(self.frame, tuple([self.target_face.left(), self.target_face.top()]), tuple([self.target_face.right(), self.target_face.bottom()]), (255, 0, 0), 1)
            
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1)

            if key == 27:
                break


            await asyncio.sleep(0.1)
            #print(self.target_face, self.info)

        #task_sleepy = asyncio.ensure_future(detect_timer(self.detect_sleepy, 10, 1.2))


        

    async def compare_face_target_monitor(self, timer, full):
        while True:
            if len(self.face_buffer) > 0:
                best_face_info = sorted(self.face_buffer, key = lambda x: (-x[0], x[1]))[0]

                if best_face_info[0] == 1:
                    face = best_face_info[2]
                    desc = best_face_info[3]

                    #self.target_face_descriptor = desc     # don't touch this
                    return face

                elif len(self.face_buffer) == full:
                    # no matching face found

                    return None


            await asyncio.sleep(timer)

    async def compare_face_target(self, face):
        if face == None:
            raise ValueError("compare_face_target: target_face is None")

        desc = await self.get_descriptor(face)
        dist = np.linalg.norm(np.array(desc) - np.array(self.target_face_descriptor))

        if dist < 0.4:
            # True
            self.face_buffer.append((1, dist, face, desc))
            return True

        else:
            # False
            self.face_buffer.append((0, dist, face, desc))
            return False

    async def get_descriptor(self, face):
        landmarks = self.landmarker(self.gray, face)
        face_descriptor = self.facerec.compute_face_descriptor(self.frame, landmarks)

        return face_descriptor


    async def init_face_descriptor(self, face):
        self.target_face_descriptor = await self.get_descriptor(face)


    def detect_expression(self, face):
        if face == None:
            raise ValueError("detect_expression: target_face is None")

        x,y,w,h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        roi_gray = self.gray[y:y+h, x:x+w]

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        except cv2.error as e:
            raise ValueError("No face detected") from e

        roi_gray = roi_gray.astype("float")
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        preds = self.exp_classifier.predict(roi_gray)[0][[3,6]]
        label = self.exp_labels[preds.argmax()]  

        return label



    def detect_gazing(self, face):
        if face == None:
            print("no target face")
            raise ValueError("detect_gazing: target_face is None")

        # isolate eye frame
        landmarks = self.landmarker(self.gray, face)

        detection_success = 0
        ratio = 0

        try:
            right_iris = self.find_iris(landmarks, range(36,42))
            right_ratio = right_iris[2]
            ratio = right_ratio
            detection_success = 1

        except ValueError as e:
            pass

        try:
            left_iris = self.find_iris(landmarks, range(42,48))
            left_ratio = left_iris[2]
            ratio += left_ratio
            detection_success += 1
        except ValueError as e:
            pass


        if detection_success > 0:
            if ratio < 0.30*detection_success:
                return "left"
            elif ratio > 0.60*detection_success:
                return "right"
            else:
                return "center"
        else:
            raise ValueError
            


    def find_iris(self, landmarks, lm_range):

        d, f = 4, 2
        eyes = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in lm_range])
        hull = np.array([np.min(eyes, axis=0), np.max(eyes, axis=0)])

        cv2.rectangle(self.frame, tuple([hull[0,0]-f, hull[0,1]-d]), tuple([hull[1,0]+f, hull[1,1]+d]), (255, 0, 0), 1)

        eye_gray = self.gray[hull[0,1]-d:hull[1,1]+d, hull[0,0]-f:hull[1,0]+f]
        eye_gray = np.clip(2.0*eye_gray.astype(np.float32)+20, 0, 255).astype(np.uint8)


        try:
            eye_gray = cv2.bilateralFilter(eye_gray, 10, 15, 15)
        except cv2.error as e:
            raise ValueError("No eye detected") from e

        cv2.imshow("eye",eye_gray)

        hist = np.histogram(eye_gray.reshape(-1), bins=20)
        hist = np.array([hist[0], hist[1][:-1]])

        thres = np.min(hist[1, np.argsort(-hist[0])][:6]) +5

        _,eye_gray = cv2.threshold(eye_gray,thres,255,cv2.THRESH_BINARY)

        cv2.imshow("eye2",eye_gray)

        contours, _ = cv2.findContours(eye_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        #print(len(contours))

        try:
            moments = cv2.moments(contours[-2])
            #print(moments)
            x = int(moments['m10'] / (moments['m00']+1e-6))
            y = int(moments['m01'] / (moments['m00']+1e-6))

            cv2.circle(self.frame, (x+hull[0,0]-f, y+hull[0,1]-d), 1, (0, 255, 0))

            return (x+hull[0,0]-f, y+hull[0,1]-d, x/eye_gray.shape[1], y/eye_gray.shape[0])
        except IndexError:
            raise ValueError

        



    def compute_eye_ratio(self, face):
        if face == None:
            raise ValueError("compute_eye_ratio: No face detected")

        landmarks = self.landmarker(self.gray, face)

        left_eye_width = landmarks_dist(42, 45, landmarks)
        left_eye_height1 = landmarks_dist(43, 47, landmarks)
        left_eye_height2 = landmarks_dist(44, 46, landmarks)
        left_eye_ratio = (left_eye_height1 + left_eye_height2) / left_eye_width*2
        
        right_eye_width = landmarks_dist(36, 39, landmarks)
        right_eye_height1 = landmarks_dist(37, 41, landmarks)
        right_eye_height2 = landmarks_dist(38, 40, landmarks)
        right_eye_ratio = (right_eye_height1 + right_eye_height2) / right_eye_width*2

        return right_eye_ratio + left_eye_ratio




def start():

    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    landmarker = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    #exp_classifier = load_model('model_v6_23.hdf5')
    exp_classifier = model_from_json(open("facial_expression_model_structure.json", "r").read())
    exp_classifier.load_weights('facial_expression_model_weights.h5')

    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    facial_detector = FacialDetector(detector=detector, landmarker=landmarker, exp_classifier=exp_classifier, facerec=facerec, cap=cap)


    return facial_detector

"""
json_file = open("handskeleton/model.json", "r") #open("face_expression_model-weights_manifest.json", "r")
loaded_model_json = json_file.read()
json_file.close()

classifier = model_from_json(loaded_model_json)
"""

def run(cap, facial_detector):
    _, frame = cap.read()
    
    facial_detector.set_frame(frame)
    result = facial_detector.detect()

    return result

if __name__ == "__main__":
    facial_detector = start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(facial_detector.run())
    

    """
    while True:
        result = run(cap, facial_detector)
        frame = facial_detector.frame

        if result != None:
            if current_exp != result:
                current_exp = result
                print(result)

            try:
                cv2.putText(frame, ' '.join(['' if x == None else x for x in result]), (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)                    #print(msg)
            except TypeError:
                pass


            cv2.imshow("Frame", frame)


            key = cv2.waitKey(1)

            if key == 27:
                break
    """

    