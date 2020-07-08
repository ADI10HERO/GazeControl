import os
import sys
import math
import time
from openvino.inference_engine import IENetwork, IECore
from main_model import MainModel
import cv2

path = 'intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'


class FaceLandmarks(MainModel):

    def __init__(self, model_path=path, device='CPU'):
        super().__init__(device=device)
        print("\n\nLoading Face Landmark Model")
        self.load_model(model_path)

    def preprocess_output(self):
        return self.outputs[0, :, 0, 0]

    def get_eyes_coordinates(self, face_crop):

        self.predict(face_crop)
        self.wait()
        self.get_output()
        output = self.preprocess_output()

        if len(output):
            right_eye = (output[0], output[1])
            left_eye = (output[2], output[3])

        return right_eye, left_eye
