#!/bin/bash

batch_size=( 1 16 32 256 512 )
pwd=$PWD;

gpuid=7
export CUDA_VISIBLE_DEVICES=$gpuid
port=$(( 7999 + $gpuid*100 ))
 
for task in imc dig; do
    stats=$pwd/stats-nvvp-$task
    mkdir -p $stats
    rm -rf $stats/*
    tcnt=1
    for batch in "${batch_size[@]}";
    do
        # gpu timing pass
        cd ../dnn/
        nvprof --devices 0 \
        --analysis-metrics \
        --output-profile $stats/log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt $tcnt &
        sid=$!
        sleep 20

        cd $pwd; 

        export glog_logtostderr=1
        export glog_log_dir=`pwd`/log
        ./img-client --hostname localhost \
        --task $task \
        --portno $port \
        --input input/${task}-input.bin \
        --num $batch \
        --flandmark data/flandmark.dat \
        --haar data/haar.xml \
        --debug 0
        sleep 5
        mv $stats/log.csv $stats/all_${task}_${batch}.csv
        kill $sid
    done
done



