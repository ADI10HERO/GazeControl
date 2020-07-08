# Gaze Control :eyes:

Control your mouse pointer with the your eyegaze! Just like you always wanted to! 

This project is the 3rd and final project for Udacity's Intel Edge AI Nanodegree program.
It not a simple task, many deep learning models need to work in harmony with each other to make it possible.

This is the **overall flow** or the architecture of the system:
<center><img src="img/pipeline.png" height="500px"></center>

## Project Set Up and Installation

**Prerequisites and Dependencies:**
  - Clone this repository
  - I suggest you make a new virtual environment, either using venv or conda
  - Activate the environment
  - Run `pip install -r requirements.txt`
  - Make sure you can run the "demo" given [here](https://docs.openvinotoolkit.org/latest/index.html) before moving ahead.
  - Download pre-trained models using the model_downloader scripts if you don't want to use the ones in the intel folder of this repository
  

*model_downloader scripts can be found at /opt/intel/deployment_tools/tools/model_downloader/downloader.py*
*example usage?* 
```sh
  python3 /opt/intel/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP16
```


**If you didn't use the default installation path, I'm sure you're smart enough to find the model_downloader script yourself :P**

## Demo
* To use webcam
` python3 src/main.py -i CAM`

* To use example video `python3 src/main.py -i bin/demo.mp4`


## Documentation

Let's talk about flags one should actually know / care about..

- `-i` This is for input path, which can be either "CAM" or path to a video file example `bin/demo.mp4`
- `-v` Verbose, example `-v True` or `-v False`, if true you wiil get information of inference times of different models for about evey 25 frames, also if set then the inference times will be logged into a file in csv format.
- `-f` Frames, Number of frames for which model info is to be printed (yes, you guessed it), default is 25 frames, it is also the count of frames to wait before any detection is done.
- `-l` If needed, this flag should be used to provide the path for an OpenVINO cpu extension.
- `-d` To specify the hardware where the models will be running. Default is CPU.
- `-pt` To state the confidence the models need to have in order to consider an actual detection

## Benchmarks

- Following benchmarks are taken on FP16 precision

|     Model     | Model load time | Avg. Inference time |
| ------------- | --------------- | ------------------- |
| Face Detector |     0.2127      |      0.1412         |
| Face Landmark |     0.0535      |      0.0067         |
| Head Pose     |     0.0696      |      0.0110         |
| Gaze Est.     |     0.0835      |      0.0102         |

- Following benchmarks are taken on FP32 precision

|     Model     | Model load time | Avg. Inference time |
| ------------- | --------------- | ------------------- |
| Face Detector |     0.2078      |      0.1409         |
| Face Landmark |     0.0501      |      0.0037         |
| Head Pose     |     0.0635      |      0.0092         |
| Gaze Est.     |     0.0796      |      0.0098         |

***Note*** : *Benchmarks were taken on CPU (intel core i5 8th Gen)*

## Results

- We can clearly see in both precisions face detector took most resources be it load time or infer time.
Rest all models are blazingly fast when compared to face detector.
- Both precision models do equally good more or less
- Even w.r.t accuracy there wasn't any noticiable difference between those models,and we can always use hardware accelarators to boost performance of lower precision models.
- Bottle Neck? One can say, it's the face detector model, solution? Can be finding a better pre-trained one or training one yourself. :D


## Note:
- Couldn't do any "standout suggestions" because of time constraints
- Any contribution appreaciated
- Steps for contributing?
    - Clone the repo
    - Make the change
    - Make a PR 
    - Please include test results like successful working screenshots / any logs in the PR
    - Get it Merged :")
