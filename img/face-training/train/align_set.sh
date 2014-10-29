#!/bin/bash

# Read params from text files
file=$1
basedir=/home/jahausw/datasets/faces/lfw/lfw_a

{
    cd .. ;
    while read line
    do
        img=$(echo $line | awk {'print $1'})
        # echo $basedir/$img
        ./run-face.sh $basedir/$img
    done
} < $file
