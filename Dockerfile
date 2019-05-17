FROM python:3.7-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    unzip \
    wget

ADD requirements.txt .

RUN pip install -r requirements.txt