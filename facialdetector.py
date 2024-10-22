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
        self.info = {"absence": "present", "expression": "neutral", "eye_dir": "center", "sleepiness": "awake"}

        self.updated = False

        self.nosetip = np.zeros((2))


        self.exp_labels = ('happy', 'neutral') #('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


    def set_frame(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    async def detect_timer(self, func, info_name, buffer_size, sleeptime):
        # notify on init, and when the result changes (after some threshold)
        try:
            result = func(self.target_face)
            buf = np.array([result])
            self.info[info_name] = result
            
            # print(result)
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
                        

                        # print(current_result)
                        self.updated = True

                    self.info[info_name] = result

            except ValueError:
                #print(info_name+ ": ValueError!")
                pass
            await asyncio.sleep(sleeptime)

    async def init_face_descriptor(self, face):
        self.target_face_descriptor = await self.get_descriptor(face)
        
        

    async def get_descriptor(self, face):
        landmarks = self.landmarker(self.gray, face)
        face_descriptor = self.facerec.compute_face_descriptor(self.frame, landmarks)

        return face_descriptor



    async def compare_face_target(self, face):
        if face == None:
            raise ValueError("compare_face_target: target_face is None")

        desc = await self.get_descriptor(face)
        try:
            dist = np.linalg.norm(np.array(desc) - np.array(self.target_face_descriptor))

            if dist < 0.4:
                # True
                self.face_buffer.append((1, dist, face, desc))
                return True

            else:
                # False
                self.face_buffer.append((0, dist, face, desc))
                return False
        except:
            return True


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

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        self.detector.setInput(blob)
        result = self.detector.forward()

        faces = []

        for i in range(result.shape[2]):
            confidence = result[0, 0, i, 2]
            if confidence > 0.5:
                box = result[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                faces.append(dlib.rectangle(x, y, x1, y1))

        return faces


    async def find_target(self, timer):

        while True:

            if False:#self.target_face != None:
                # local search around previous target face
                face = self.target_face
                d = 300
                hrange = range(max(face.top()-d,0), min(face.bottom()+d,self.gray.shape[0]))
                wrange = range(max(face.left()-d,0), min(face.right()+d,self.gray.shape[1]))

                faces = self.detect_faces(self.frame[hrange][:,wrange]) #self.detector(self.gray[hrange][:,wrange])

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

                #faces = self.detector(self.gray)

                faces = self.detect_faces(self.frame)
                if len(faces) >= 1: #len(faces) == 1:
                    face = faces[0]
                    self.target_face = face

                    try:
                        self.target_face_descriptor = await get_descriptor(face)
                    except:
                        pass
                

                elif len(faces) > 0:

                    tasks = []

                    for face in faces:
                        tasks.append(asyncio.ensure_future(self.compare_face_target(face)))

                    task_monitor = asyncio.ensure_future(self.compare_face_target_monitor(0.33*timer, len(faces)))
                    self.target_face = await task_monitor

                    for task in tasks:
                        if not task.done():
                            task.cancel()
                else:
                    self.target_face = None

            await asyncio.sleep(timer)


    def detect_sleepy(self, face):
        if self.mean_eye_ratio == None:
            raise ValueError

        try:
            y=self.compute_eye_ratio(face)
            # print(y)
        except:
            raise ValueError("detect_sleepy: No face detected")

        return "sleepy" if y<0.8*self.mean_eye_ratio else "awake"


    async def init_mean_eye_ratio(self,  buffer_size, timer):
        s = 0
        i = 0

        while i < buffer_size:
            frame = self.cap.read()[1]
            self.set_frame(frame)

            try:
                s += self.compute_eye_ratio(self.target_face)
                i += 1 
            except:
                pass

            await asyncio.sleep(timer)

        # print(s/buffer_size)
        self.mean_eye_ratio = s / buffer_size


    async def run(self):
        task_target = asyncio.ensure_future(self.find_target(0.1))
        
        task_absence = asyncio.ensure_future(self.detect_timer(lambda x: "absence" if x==None else "present", 
                                                                "absence", 3, 1))

        task_exp = asyncio.ensure_future(self.detect_timer(self.detect_expression, "expression", 3, 0.2))

        task_gazing = asyncio.ensure_future(self.detect_timer(self.detect_headpose, "eye_dir", 4, 0.4))

        sleepy_init = asyncio.ensure_future(self.init_mean_eye_ratio(10, 0.2))

        task_sleepy = asyncio.ensure_future(self.detect_timer(self.detect_sleepy,"sleepiness", 3, 1))
        
        while True:
            frame = self.cap.read()[1]
            self.set_frame(frame)
                # print(result)

            if self.target_face != None:
                cv2.rectangle(self.frame, tuple([self.target_face.left(), self.target_face.top()]), tuple([self.target_face.right(), self.target_face.bottom()]), (255, 0, 0), 1)
                landmarks = self.landmarker(self.gray, self.target_face)

                for i in [33, 8, 36, 45, 60, 64]:
                    cv2.circle(self.frame, (landmarks.part(i).x, landmarks.part(i).y), 3, (255,0,0), 10)
            
            cv2.circle(self.frame, tuple(self.nosetip.astype(int)), 3, (0,255,0), 10)

            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1)

            if key == 27:
                break


            # print(self.info)

            await asyncio.sleep(0.05)
            #print(self.target_face, self.info)

        

    


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

    # reference : https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    def detect_headpose(self, face):
        if face == None:
            raise ValueError("detect_headpose: No face detected")

        landmarks = self.landmarker(self.gray, face)

        image_points = np.array([
                            [landmarks.part(33).x, landmarks.part(33).y],    # nose tip
                            [landmarks.part(8).x, landmarks.part(8).y],     # chin
                            [landmarks.part(36).x, landmarks.part(36).y],   # left eye left corner
                            [landmarks.part(45).x, landmarks.part(45).y],   # right eye right corner
                            [landmarks.part(60).x, landmarks.part(60).y],   # left mouth corner
                            [landmarks.part(64).x, landmarks.part(64).y]    # right mouth corner
                            ], dtype="double")
        model_points = np.array([
                            [0.0, 0.0, 0.0],             # Nose tip
                            [0.0, -330.0, -65.0],        # Chin
                            [-225.0, 170.0, -135.0],     # Left eye left corner
                            [225.0, 170.0, -135.0],      # Right eye right corne
                            [-150.0, -150.0, -125.0],    # Left Mouth corner
                            [150.0, -150.0, -125.0]      # Right mouth corner
                            ])


        size = self.gray.shape
        
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], 
                         dtype = "double")

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        nose_end_point2D = nose_end_point2D.squeeze()

        self.nosetip = np.copy(nose_end_point2D)

        nose_end_point2D -= np.array([face.left()*0.5+face.right()*0.5, face.bottom()*0.5+face.top()*0.5])
        w = face.right() - face.left()
        # print(nose_end_point2D)

        if nose_end_point2D[0] < -w*0.7:
            return "left"
        elif nose_end_point2D[0] > w*0.7:
            return "right"
        else:
            return "center"

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
                return "right"
            elif ratio > 0.60*detection_success:
                return "left"
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

    #detector = dlib.get_frontal_face_detector()
    modelFile = "models/opencv_face_detector.caffemodel"
    configFile = "models/opencv_face_detector.prototxt.txt"

    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    landmarker = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    #exp_classifier = load_model('model_v6_23.hdf5')

    #exp_classifier = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
    #exp_classifier.load_weights('models/facial_expression_model_weights.h5')
    
    exp_classifier = model_from_json(open("models/model.json", "r").read())
    exp_classifier.load_weights('models/weights.h5')

    facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

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
    print("test1")
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

    