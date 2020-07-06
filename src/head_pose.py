import os
import sys
import math
import time
from openvino.inference_engine import IENetwork, IECore
from main_model import MainModel
import cv2

path = 'intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'

class HeadPose(MainModel):

    def __init__(self, model_path=path, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)


    def get_output(self, request_id=0):
        self.outputs = self.net_plugin.requests[request_id].outputs
        return self.outputs


    def preprocess_output(self):
        return [self.outputs['angle_y_fc'][0,0], self.outputs['angle_p_fc'][0,0], self.outputs['angle_r_fc'][0,0]]


    def get_headpose_angles(self, face_crop):

        self.predict(face_crop)
        self.wait()
        self.get_output()
        output = self.preprocess_output()

        return output
