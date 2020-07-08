"""
Template of the main file took from Project1 - Footfall
https://github.com/ADI10HERO/FootFall/blob/master/main.py 
"""

import cv2
import time
import csv

from face_detector import FaceDetector
from face_landmarks import FaceLandmarks
from head_pose import HeadPose
from gaze_estimation import GazeEst

from mouse_controller import MouseController
from argparse import ArgumentParser


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-f", "--frames", type=float, default=25,
                        help="Number of frames to actually run the models on."
                        "(25 by default)")
    parser.add_argument("-v", "--verbose", type=bool, default=False,
                        help="Verbosity of the program for printing info on the screen and writing to csv"
                        "(False by default)")

    return parser


def get_eyes_crops(face_crop, right_eye, left_eye, relative_eye_size=0.20):

    crop_w = face_crop.shape[1]
    crop_h = face_crop.shape[0]

    x_right_eye = right_eye[0]*crop_w
    y_right_eye = right_eye[1]*crop_h
    x_left_eye = left_eye[0]*crop_w
    y_left_eye = left_eye[1]*crop_h

    relative_eye_size_x = crop_w*relative_eye_size
    relative_eye_size_y = crop_h*relative_eye_size

    right_eye_dimensions = [int(y_right_eye-relative_eye_size_y/2), int(y_right_eye+relative_eye_size_y/2),
                            int(x_right_eye-relative_eye_size_x/2), int(x_right_eye+relative_eye_size_x/2)]

    left_eye_dimensions = [int(y_left_eye-relative_eye_size_y/2), int(y_left_eye+relative_eye_size_y/2),
                           int(x_left_eye-relative_eye_size_x/2), int(x_left_eye+relative_eye_size_x/2)]

    right_eye_crop = face_crop[right_eye_dimensions[0]:right_eye_dimensions[1],
                               right_eye_dimensions[2]:right_eye_dimensions[3]]

    left_eye_crop = face_crop[left_eye_dimensions[0]:left_eye_dimensions[1],
                              left_eye_dimensions[2]:left_eye_dimensions[3]]

    return right_eye_crop, left_eye_crop, right_eye_dimensions, left_eye_dimensions


def get_models(args):
    facedetector = FaceDetector(device=args.device)
    facelm = FaceLandmarks(device=args.device)
    headpose = HeadPose(device=args.device)
    gaze_est = GazeEst()
    return facedetector, facelm, headpose, gaze_est


def start_gc(args, facedetector, facelm, headpose, gaze):

    mouse_controller = MouseController(precision='high', speed='fast')

    inf_time_f = []
    inf_time_lm = []
    inf_time_h = []
    inf_time_g = []

    if args.input != 'CAM':
        try:
            input_stream = cv2.VideoCapture(args.input)
            length = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

            if length > 1:
                single_image_mode = False
            else:
                single_image_mode = True

        except:
            print(
                'Not supported image or video file format. Please pass a supported one.')
            exit(0)

    else:
        input_stream = cv2.VideoCapture(0)
        single_image_mode = False

    if not single_image_mode:
        count = 0
        while(input_stream.isOpened()):

            # Read the next frame:
            flag, frame = input_stream.read()

            if not flag:
                break

            if count % args.frames == 0:
                start = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                face_crop, detection = facedetector.get_face_crop(frame, args)
                finish_f = time.time()
                time_f = finish_f-start

                if args.verbose:
                    print("Face detection took {} seconds.".format(
                        time_f))
                    inf_time_f.append(time_f)

                right_eye, left_eye = facelm.get_eyes_coordinates(face_crop)
                right_eye_crop, left_eye_crop, right_eye_coords, left_eye_coords = get_eyes_crops(
                    face_crop, right_eye, left_eye)
                finish_ed = time.time()
                time_ed = finish_ed-finish_f

                if args.verbose:
                    print("Eyes detection took {} seconds.".format(
                        time_ed))
                    inf_time_lm.append(time_ed)

                headpose_angles = headpose.get_headpose_angles(face_crop)
                finish_h = time.time()
                time_h = finish_h-finish_ed

                if args.verbose:
                    print("Pose detection took {} seconds.".format(
                        time_h))
                    inf_time_h.append(time_h)

                (x_movement, y_movement), gaze_vector = gaze.get_gaze(
                    right_eye_crop,
                    left_eye_crop,
                    headpose_angles
                )
                time_g = time.time()-finish_h

                if args.verbose:
                    print("Gaze detection took {} seconds.".format(
                        time_g))
                    inf_time_g.append(time_g)

                frame = cv2.rectangle(frame,
                                      (detection[0], detection[1]),
                                      (detection[2], detection[3]),
                                      color=(0, 255, 0), thickness=5)

                right_eye_coords = [right_eye_coords[0]+detection[1], right_eye_coords[1]+detection[1],
                                    right_eye_coords[2]+detection[0], right_eye_coords[3]+detection[0]]

                left_eye_coords = [left_eye_coords[0]+detection[1], left_eye_coords[1]+detection[1],
                                   left_eye_coords[2]+detection[0], left_eye_coords[3]+detection[0]]

                frame = cv2.rectangle(frame,
                                      (right_eye_coords[2],
                                       right_eye_coords[1]),
                                      (right_eye_coords[3],
                                       right_eye_coords[0]),
                                      color=(255, 0, 0),
                                      thickness=5)

                frame = cv2.rectangle(frame,
                                      (left_eye_coords[2],
                                       left_eye_coords[1]),
                                      (left_eye_coords[3],
                                       left_eye_coords[0]),
                                      color=(255, 0, 0),
                                      thickness=5)

                # Right eye:
                x_r_eye = int(right_eye[0]*face_crop.shape[1]+detection[0])
                y_r_eye = int(right_eye[1]*face_crop.shape[0]+detection[1])
                x_r_shift, y_r_shift = int(
                    x_r_eye+gaze_vector[0]*100), int(y_r_eye-gaze_vector[1]*100)

                # Left eye:
                x_l_eye = int(left_eye[0]*face_crop.shape[1]+detection[0])
                y_l_eye = int(left_eye[1]*face_crop.shape[0]+detection[1])
                x_l_shift, y_l_shift = int(
                    x_l_eye+gaze_vector[0]*100), int(y_l_eye-gaze_vector[1]*100)

                frame = cv2.arrowedLine(
                    frame, (x_r_eye, y_r_eye), (x_r_shift, y_r_shift), (0, 0, 255), 2)
                frame = cv2.arrowedLine(
                    frame, (x_l_eye, y_l_eye), (x_l_shift, y_l_shift), (0, 0, 255), 2)

                cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Output', 600, 400)
                cv2.imshow('Output', frame)

                if args.verbose:
                    print()

                mouse_controller.move(x_movement, y_movement)
            count = count + 1

        input_stream.release()
        if args.verbose:
            with open('inference.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Face Detector', 'Eyes Detector',
                                 'Headpose Detector', 'Gaze Detector'])
                for i in range(len(inf_time_f)):
                    writer.writerow([inf_time_f[i], inf_time_lm[i],
                                     inf_time_h[i], inf_time_g[i]])

    cv2.destroyAllWindows()


def main():
    args = build_argparser().parse_args()
    facedetector, facelm, headpose, gaze_est = get_models(args)
    start_gc(args, facedetector, facelm, headpose, gaze_est)


if __name__ == '__main__':
    main()
