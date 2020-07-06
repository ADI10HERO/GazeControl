'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import math
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


def handle_image(input_image, width=60, height=60):
    """
    Function to preprocess input image and return it in a shape accepted by the model.
    Default arguments are set for facial landmark model requirements.
    """
    preprocessed_image = cv2.resize(input_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2,0,1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)

    return preprocessed_image


class MainModel:
    '''
    Main Super class, most parts referenced from project-1 
    '''
    def __init__(self, device):
        self.device = device
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.outputs = None


    def load_model(self, model, cpu_extension=None, plugin=None):
        start_time = time.time()

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        if not plugin:
            print("Initializing plugin")
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in self.device:
            self.plugin.add_extension(cpu_extension, "CPU")

        print("Reading IR")
        self.net = IENetwork(model=model_xml, weights=model_bin)

        print("Loading IR to the plugin")
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.net, "CPU")
            not_supported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

            if not_supported_layers:
                print("Some layers are not supported, exiting")
                sys.exit(1)

        self.net_plugin = self.plugin.load_network(network=self.net, device_name=self.device)
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        finish_time = time.time()
        print("time taken",finish_time-start_time)
        return self.plugin

    def predict(self, image, request_id=0):

        preprocessed_image = self.preprocess_input(image)
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, 
                                                                inputs={self.input_blob: preprocessed_image})

        return self.net_plugin

    def check_model(self):
        pass

    def preprocess_input(self, image):

        input_shape = self.net.inputs[self.input_blob].shape
        preprocessed_image = handle_image(image, input_shape[3], input_shape[2])

        return preprocessed_image

    def wait(self, request_id=0):
        status = self.net_plugin.requests[request_id].wait(-1)

        return status

    def get_output(self, request_id=0):

        self.outputs = self.net_plugin.requests[request_id].outputs[self.out_blob]

        return self.outputs

