#!/bin/sh

sed -i -e "s/batch_size:.*/batch_size:\ ${2}/" input/$1-inputnet.prototxt
