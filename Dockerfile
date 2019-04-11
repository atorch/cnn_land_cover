FROM python:3.7-slim

RUN apt-get update && apt-get install -y \
    gdal-bin \
    unzip \
    wget

RUN pip install \
    fiona \
    keras \
    numpy \
    pyproj \
    rasterio \
    shapely \
    sklearn \
    tensorflow