FROM tensorflow/tensorflow:latest

ENV INPUT_PATH=data/images/original/Trial1_Right-051.jpeg

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN apt-get install -y git python
RUN pip install --upgrade pip
RUN pip install pandas keras pillow matplotlib opencv-python==4.1.2.30

RUN git clone https://github.com/apurl1/tiny-yolo.git
WORKDIR /tiny-yolo
ADD ${INPUT_PATH} ${INPUT_PATH}

CMD git pull && python yolo_video.py --image
