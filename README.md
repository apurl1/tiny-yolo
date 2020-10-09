# tiny-yolo

## Overview
This code is a bare minimum version of [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3). It only runs Tini YOLOv3 on images and videos and does not provide support for custom training.

The script returns a list of (label, confidence, bounding box coordinates) items. There is also an option to write these to a csv file and a script that provides some post-processing functionality for the results.

## Usage
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
  --csv              path to csv output file
```
Example: `python yolo_video.py --model model_data/yolo-tiny.h5 --anchors model_data/tiny_yolo_anchors.txt --input data/videos/Trial1/1_right.mp4 --output results/vidoes/1_right.mp4`
