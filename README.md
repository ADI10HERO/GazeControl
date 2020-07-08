# Computer Pointer Controller

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


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Note:
- Couldn't do any "standout suggestions" because of time constraints
- Any contribution appreaciated
- Steps for contributing?
    - Clone the repo
    - Make the change
    - Make a PR 
    - Please include test results like successful working screenshots / any logs in the PR
    - Get it Merged :")
