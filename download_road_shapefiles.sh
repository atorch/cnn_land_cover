#!/bin/bash

mkdir -p roads

cd roads

for county in 17011 55023 55103
do
    file=tl_2017_${county}_roads.zip
    wget ftp://ftp2.census.gov/geo/tiger/TIGER2017/ROADS/$file --no-clobber
    unzip -o $file
done
