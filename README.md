# CNN for Land Cover

```bash
./scripts/download_cdl.sh
./scripts/download_county_shapefile.sh
```

```bash
export DOCKER_TAG=cnn_land_cover_docker
sudo docker build ~/cnn_land_cover --tag=$DOCKER_TAG
sudo docker run -it -v ~/cnn_land_cover:/home/cnn_land_cover $DOCKER_TAG bash
cd /home/cnn_land_cover
python src/annotate_naip_scenes.py
python src/fit_model.py
```

# TODO

* [ ] Download script for the NAIP scenes in [model_config.yml](config/model_config.yml)
* [ ] Env var for year, use it in all download scripts
* [ ] Qix spatial index files for shapefiles
* [ ] More objectives: buildings, semantic segmentation...
* [ ] GPU
* [ ] Test set confusion matrices
* [ ] Tensorboard
* [ ] Tune dropout probability, number of filters, number of blocks
* [ ] Visualizations

# Datasets

[Census TIGER Shapefiles](https://www.census.gov/geo/maps-data/data/tiger-line.html) (roads and counties)

[Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/) (land cover raster, visualize it [here](https://nassgeodata.gmu.edu/CropScape/))

[Microsoft building footprints](https://github.com/microsoft/USBuildingFootprints)

[National Agriculture Imagery Program](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) (four-band aerial imagery, originally downloaded from [USGS Earth Explorer](https://earthexplorer.usgs.gov/?))

The Cropland Data Layer (CDL) and TIGER road shapefiles are used to programmatically generate
labels for NAIP images. See [sample_images](sample_images) for sample input images along
with their labels (one label per objective).

# Models

[Fully convolutional neural network](cnn.py) with four objectives:

```
Classification report for is_majority_forest:
              precision    recall  f1-score   support

           0       0.99      0.98      0.98       506
           1       0.89      0.93      0.91        94

    accuracy                           0.97       600
   macro avg       0.94      0.95      0.94       600
weighted avg       0.97      0.97      0.97       600

Classification report for has_roads:
              precision    recall  f1-score   support

           0       0.81      0.93      0.86       269
           1       0.93      0.82      0.88       331

    accuracy                           0.87       600
   macro avg       0.87      0.88      0.87       600
weighted avg       0.88      0.87      0.87       600

Classification report for modal_land_cover:
              precision    recall  f1-score   support

    corn_soy       0.85      0.74      0.79       191
   developed       0.85      0.93      0.89       149
      forest       0.82      0.86      0.84       112
       other       0.53      0.16      0.25        56
     pasture       0.23      0.45      0.30        51
       water       0.90      0.95      0.93        20
    wetlands       0.71      0.57      0.63        21

    accuracy                           0.73       600
   macro avg       0.70      0.66      0.66       600
weighted avg       0.76      0.73      0.73       600

Classification report for pixels:
              precision    recall  f1-score   support

       other       0.72      0.86      0.79  19218490
        road       0.70      0.04      0.08    875662
      forest       0.77      0.80      0.78   6566224
    corn_soy       0.88      0.63      0.73  11372577
       water       0.86      0.88      0.87   1288647

    accuracy                           0.77  39321600
   macro avg       0.78      0.64      0.65  39321600
weighted avg       0.78      0.77      0.76  39321600
```