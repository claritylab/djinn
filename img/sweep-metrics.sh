#!/bin/bash

metrics="achieved_occupancy,sm_efficiency_instance,ipc_instance,l2_utilization,alu_fu_utilization"
agg='off'
batch_size=( 1 2 4 8 16 32 64 128 256 512 )
pwd=$PWD;

gpuid=1
export CUDA_VISIBLE_DEVICES=$gpuid
port=$(( 7999 + $gpuid*100 ))
 
for task in imc dig face; do
    stats=$pwd/stats-$agg-$task
    mkdir -p $stats
    rm -rf $stats/*
    tcnt=1
    for batch in "${batch_size[@]}";
    do
        echo $task $batch
        cd ../dnn/
        if [ "$task" == "face" ]; then
            ./change_batch.sh face $batch
        fi
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv $stats/timing_${task}_${batch}.csv --threadcnt $tcnt &
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

        kill $sid
        cd ../dnn/
        nvprof --devices 0 \
        --aggregate-mode $agg \
        --metrics $metrics \
        --csv \
        --log-file $stats/log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt 1 &
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
        mv $stats/log.csv $stats/prof_${task}_${batch}.csv
        kill $sid

        # gpu timing pass
        cd ../dnn/
        nvprof --devices 0 \
        --aggregate-mode $agg \
        --print-gpu-summary \
        --csv \
        --log-file $stats/log.csv \
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
