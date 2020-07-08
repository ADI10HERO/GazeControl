import os
import sys
import math
import time
from openvino.inference_engine import IENetwork, IECore
from main_model import MainModel
import cv2


path = 'intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'


class FaceDetector(MainModel):

    def __init__(self, model_path=path, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)

    def preprocess_output(self):

        return self.outputs[0, 0]

    def get_face_crop(self, frame, args):

        threshold = args.prob_threshold
        self.predict(frame)
        self.wait()
        self.get_output()
        output = self.preprocess_output()

        if len(output):
            detection = []
            for o in output:
                if o[2] > threshold:
                    xmin, ymin, xmax, ymax = o[3], o[4], o[5], o[6]
                    detection.append([xmin, ymin, xmax, ymax])

        detection = detection[0]

        w = frame.shape[1]
        h = frame.shape[0]
        detection = [int(detection[0]*w), int(detection[1]*h),
                     int(detection[2]*w), int(detection[3]*h)]

        return frame[detection[1]:detection[3], detection[0]:detection[2]], detection
