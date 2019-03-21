# CNN for Land Cover

```bash
./download_county_shapefile.sh
./download_cdl.sh
```

```bash
export DOCKER_TAG=cnn_land_cover_docker
sudo docker build ~/cnn_land_cover --tag=$DOCKER_TAG
sudo docker run -it -v ~/cnn_land_cover:/home/cnn_land_cover $DOCKER_TAG bash
cd /home/cnn_land_cover
python annotate_naip_scenes.py
python fit_model.py
```

# TODO

* [ ] Docker
* [ ] Multi-objective model (roads, buildings, ...)
* [ ] Download script for [naip_scenes.txt](naip_scenes.txt)

# Datasets

[Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/) (land cover raster)

[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) (four-band aerial imagery)

# Models
