#!/bin/bash

mkdir -p roads

cd roads

for county in 17011 17019 17053 17103 17113 17147 19101 27129 27173 55023 55103
do
    base=tl_2017_${county}_roads
    zipfile=${base}.zip
    shpfile=${base}.shp
    wget ftp://ftp2.census.gov/geo/tiger/TIGER2017/ROADS/${zipfile} --no-clobber
    unzip -o $zipfile
    ogrinfo -sql "CREATE SPATIAL INDEX ON ${base}" ${shpfile}
done
