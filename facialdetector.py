import cv2
import numpy as np
import dlib
import math
import time
import json

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, model_from_json
from keras.preprocessing.image import img_to_array

def euc_dist(x1, y1, x2, y2):
    return math.sqrt((abs(x1-x2))**2+(abs(y1-y2))**2)

def landmarks_dist(n1, n2, landmarks):
    return euc_dist(landmarks.part(n1).x, landmarks.part(n1).y, landmarks.part(n2).x, landmarks.part(n2).y)

class FacialDetector:

    def __init__(self, detector, landmarker, exp_classifier, facerec):
        self.detector = detector
        self.landmarker = landmarker
        self.exp_classifier = exp_classifier
        self.facerec = facerec 

        self.gray = None
        #self.faces = None
        self.frame = None

        self.target_face = None

        self.absence_time = -1
        self.absence_thres_time = 2

        self.sleepy_thres_frames = 600
        self.mean_eye_ratio = 0
        self.sleepy_buffer = -np.ones(6000)   # save recent 10 frames here
        self.sleepy_buffer_pt = 0   # next location pointer

        self.exp_labels = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


    def set_frame(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

    def detect(self):
        # TODO: implement facial recognition, and only detect a specific face
        faces = self.detector(self.gray)


        if len(faces)  == 0:
            return tuple([self.detect_absence(), None, None])

        else:
            for face in faces:
                if self.target_face == None:
                    self.init_face_descriptor(face)
                    #return None


                #if not self.compare_face_target(face):
                #    continue

                self.absence_time = -1

                # display rectangle
                x,y,w,h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

                sleepy = self.detect_sleepy(face)
                return tuple(['present', self.detect_expression(face) if not sleepy else "sleepy", self.detect_gazing(face)])


    def detect_absence(self):
        if self.absence_time == -1:
            self.absence_time = time.time()
            return "unseen"

        else:
            current_time = time.time()
            if current_time >= self.absence_time + self.absence_thres_time:
                return "absence"
            else:
                return "unseen"

    def get_descriptor(self, face):
        landmarks = self.landmarker(self.gray, face)
        face_descriptor = self.facerec.compute_face_descriptor(self.frame, landmarks)

        return face_descriptor

    def init_face_descriptor(self,face):
        self.target_face = self.get_descriptor(face)

    def compare_face_target(self, face):
        #print(np.array(self.get_descriptor(face)))

        dist = np.linalg.norm(np.array(self.get_descriptor(face)) - np.array(self.target_face))

        if dist < 0.6:
            return True
        else:
            return False

    def detect_expression(self, face):
        x,y,w,h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        roi_gray = self.gray[y:y+h, x:x+w]

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        except:
            return None

        roi_gray = roi_gray.astype("float")
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        preds = self.exp_classifier.predict(roi_gray)[0]
        label = self.exp_labels[preds.argmax()]  

        return label


    def detect_sleepy(self, face):
        self.record_sleepy(face)

        if self.mean_eye_ratio == 0:
            self.init_mean_eye_ratio()
            return False

        else:
            # if you closed your eye for more than (sleepy_thres_frames)...
            slept = np.sum(self.sleepy_buffer[self.sleepy_buffer_pt - self.sleepy_thres_frames :self.sleepy_buffer_pt] < self.mean_eye_ratio/1.4)

            if slept == self.sleepy_thres_frames:
                return True
            else:
                return False


    def detect_gazing(self, face):
        # isolate eye frame
        landmarks = self.landmarker(self.gray, face)

        right_iris = self.find_iris(landmarks, range(36,42))
        left_iris = self.find_iris(landmarks, range(42,48))

        detection_success = 0
        ratio = 0

        try:
            right_ratio = right_iris[2]
            ratio = right_ratio
            detection_success = 1
        except TypeError:
            pass
        
        try:
            left_ratio = left_iris[2]
            ratio = ratio + left_ratio
            detection_success = detection_success + 1
        except TypeError:
            pass

        if detection_success != 0:
            if ratio < 0.35*detection_success:
                return "left"
            elif ratio > 0.65*detection_success:
                return "right"
            else:
                return "center"
            


    def find_iris(self, landmarks, lm_range):
        d, f = 5, 2
        eyes = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in lm_range])
        hull = np.array([np.min(eyes, axis=0), np.max(eyes, axis=0)])

        cv2.rectangle(self.frame, tuple([hull[0,0]-f, hull[0,1]-d]), tuple([hull[1,0]+f, hull[1,1]+d]), (255, 0, 0), 1)

        eye_gray = self.gray[hull[0,1]-d:hull[1,1]+d, hull[0,0]-f:hull[1,0]+f]
        eye_gray = cv2.bilateralFilter(eye_gray, 10, 15, 15)
        cv2.imshow("eye",eye_gray)

        hist = np.histogram(eye_gray.reshape(-1), bins=20)
        thres = np.min(sorted(hist, key= lambda x: -x[0])[1][:3]) + 5

        #print(thres)
        

        _,eye_gray = cv2.threshold(eye_gray,thres,255,cv2.THRESH_BINARY)

        cv2.imshow("eye2",eye_gray)

        contours, _ = cv2.findContours(eye_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            #circles = cv2.HoughCircles(eye_gray, cv2.HOUGH_GRADIENT, 1, 20,param1=50,param2=5,minRadius=0, maxRadius=int(eye_gray.shape[1]*0.8))
            #circles = np.uint16(np.around(circles))

            moments = cv2.moments(contours[-2])
            x = int(moments['m10'] / moments['m00'])
            y = int(moments['m01'] / moments['m00'])

            #x = circles[0,0,0]
            #y = circles[0,0,1]

            cv2.circle(self.frame, (x+hull[0,0]-f, y+hull[0,1]-d), 1, (0, 255, 0))
            return (x+hull[0,0]-f, y+hull[0,1]-d, x/eye_gray.shape[1], y/eye_gray.shape[0])

        except (IndexError, ZeroDivisionError, TypeError):
            return None


    def init_mean_eye_ratio(self):
        if np.sum(self.sleepy_buffer > 0) == 10:
            self.mean_eye_ratio = np.mean(self.sleepy_buffer[:10])
            #print(self.mean_eye_ratio)

        


    def record_sleepy(self, face):
        self.sleepy_buffer[self.sleepy_buffer_pt] = self.compute_eye_ratio(face)
        self.sleepy_buffer_pt = (self.sleepy_buffer_pt + 1) % self.sleepy_buffer.shape[0]


    def compute_eye_ratio(self, face):
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

    facial_detector = FacialDetector(detector=detector, landmarker=landmarker, exp_classifier=exp_classifier, facerec=facerec)


    return cap, facial_detector

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
    cap, facial_detector = start()
    current_exp = ''

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
    