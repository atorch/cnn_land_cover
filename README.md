# CNN for Land Cover

```bash
./scripts/download_cdl.sh
./scripts/download_county_shapefile.sh
./scripts/download_road_shapefiles.sh
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

* [ ] Download script for [naip_scenes.txt](naip_scenes.txt)
* [ ] Env var for year, use it in all download scripts
* [ ] Qix spatial index files for shapefiles
* [ ] More objectives: buildings, semantic segmentation...

# Datasets

[Census TIGER Shapefiles](https://www.census.gov/geo/maps-data/data/tiger-line.html) (roads and counties)

[Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/) (land cover raster, visualize it [here](https://nassgeodata.gmu.edu/CropScape/))

[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) (four-band aerial imagery, originally downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/?))

# Models
