import os
import sys
import math
import time
from openvino.inference_engine import IENetwork, IECore
from main_model import MainModel
import cv2

path = 'intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml'


def handle_image(input_image, width=60, height=60):

    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2, 0, 1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)

    return preprocessed_image


class GazeEst(MainModel):
    def __init__(self, model_path=path, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)

        self.input_name = [i for i in self.net_plugin.inputs.keys()]
        self.input_shape = self.net_plugin.inputs[self.input_name[1]].shape

    def predict(self, left_eye_crop, right_eye_crop, headpose_angles, request_id=0):
        '''
        Function to make an async inference request.
        '''
        preprocessed_left_eye_crop, preprocessed_right_eye_crop = self.preprocess_input(
            left_eye_crop, right_eye_crop)

        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id,
                                                                inputs={'head_pose_angles': headpose_angles,
                                                                        'left_eye_image': preprocessed_left_eye_crop,
                                                                        'right_eye_image': preprocessed_right_eye_crop})

        return self.net_plugin

    def preprocess_input(self, left_eye_crop, right_eye_crop):
        '''
        Function to preprocess input image according to model requirement.
        '''

        preprocessed_left_eye_crop = handle_image(
            left_eye_crop, self.input_shape[3], self.input_shape[2])
        preprocessed_right_eye_crop = handle_image(
            right_eye_crop, self.input_shape[3], self.input_shape[2])

        return preprocessed_left_eye_crop, preprocessed_right_eye_crop

    def get_output(self, request_id=0):

        self.outputs = self.net_plugin.requests[request_id].outputs
        return self.outputs

    def preprocess_output(self, headpose_angles):

        angle_r_fc = headpose_angles[2]
        roll_cosine = math.cos(angle_r_fc*math.pi/180.0)
        roll_sine = math.sin(angle_r_fc*math.pi/180.0)

        gaze_vector = self.outputs['gaze_vector'][0]

        x_movement = gaze_vector[0] * roll_cosine + gaze_vector[1] * roll_sine
        y_movement = -gaze_vector[0] * roll_sine + gaze_vector[1] * roll_cosine

        return (x_movement, y_movement), gaze_vector

    def get_gaze(self, right_eye_crop, left_eye_crop, headpose_angles):

        try:
            self.predict(left_eye_crop, right_eye_crop, headpose_angles)
            self.wait()
            self.get_output()

            (x_movement, y_movement), gaze_vector = self.preprocess_output(
                headpose_angles)

        except:
            return (0, 0), 0

        return (x_movement, y_movement), gaze_vector
