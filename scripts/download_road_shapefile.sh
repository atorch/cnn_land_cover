#!/bin/bash

mkdir -p roads

cd roads

base=tl_2017_$1_roads
zipfile=${base}.zip
shpfile=${base}.shp
wget ftp://ftp2.census.gov/geo/tiger/TIGER2017/ROADS/${zipfile} --no-clobber
unzip -o $zipfile
ogrinfo -sql "CREATE SPATIAL INDEX ON ${base}" ${shpfile}
