#!/bin/bash

cd ./buildings

for state in Iowa Illinois Minnesota Wisconsin
do
    wget https://usbuildingdata.blob.core.windows.net/usbuildings-v1-1/$state.zip --no-clobber
    unzip -o $state.zip
done
