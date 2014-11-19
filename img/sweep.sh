#!/bin/bash

#sweep img batch size

gpuid=6
hostname=localhost
portno=$(( 8000+$gpuid*100))

for t in face; do
    for i in 1 16 32 64 128 256 512 1024 2048; do
    # for i in 2048; do
        echo $t $i
        export GLOG_logtostderr=1
        export GLOG_log_dir=`pwd`/log
        ./img-client --hostname localhost \
        --task $t \
        --portno $portno \
        --queries 1 \
        --num $i \
        --input input/$1-input.bin \
        --flandmark data/flandmark.dat \
        --haar data/haar.xml \
        --debug 0
        sleep 1
    done
done
