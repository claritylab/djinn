#!/bin/sh

sed -i -e "s/batch_size:.*/batch_size:\ ${1}/" input/*-inputnet.prototxt
