FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && apt-get install -y \
    gdal-bin \
    unzip \
    wget

WORKDIR /home/cnn_land_cover

ADD requirements.txt .

RUN pip install -r requirements.txt