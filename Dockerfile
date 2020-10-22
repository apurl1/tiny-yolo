FROM tensorflow/tensorflow:latest

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python
RUN pip install --upgrade pip
RUN pip install pandas

RUN git clone https://github.com/apurl1/tiny-yolo.git
ADD *.py ./

CMD python yolo_video.py --model model_data/yolo-tiny.h5 --anchors model_data/tiny_yolo_anchors.txt --input data/videos/Trial1/1_right.mp4 --output results/vidoes/1_right.mp4
