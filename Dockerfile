FROM tensorflow/tensorflow:2.3.0-gpu

RUN apt-get update && apt-get install -y \
    gdal-bin \
    unzip \
    wget

WORKDIR /home/cnn_land_cover

ADD requirements.txt .

RUN pip install -r requirements.txt