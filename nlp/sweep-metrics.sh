#!/bin/bash

input="input/batch-input.txt"
touch $input
sentence="Rockwell International Corp. 's Tulsa unit said it signed a tentative agreement extending its contract with Boeing Co. to provide structural parts for Boeing 's 747 jetliners . "

metrics="achieved_occupancy,sm_efficiency,ipc,stall_memory_throttle,dram_utilization"

batch_size=( 1 2 4 8 16 32 64 128 256 512 1024 2048)
pwd=$PWD;

gpuid=6
export CUDA_VISIBLE_DEVICES=$gpuid
port=$(( 7999 + $gpuid*100 ))
stats=$pwd/stats/
mkdir -p $stats
rm -rf $stats/*
 
for task in pos ner; do
    tcnt=1
    for batch in "${batch_size[@]}";
    do
        echo $task $batch
        # timing
        rm -rf $input
        for i in `seq 1 $batch`;
        do
            echo -n "$sentence" >> $input;
        done
        if [ "$task" == "chk" ]; then
            tcnt=2
        elif [ "$task" == "srl" ]; then
            tcnt=4
        fi
        cd ../dnn/
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv $stats/timing_${task}_${batch}.csv --threadcnt $tcnt &
        sid=$!
        sleep 20

        cd $pwd; 

        ./run-nlp.sh $task $port $input

        # metrics
        kill $sid
        cd ../dnn/
        nvprof --devices 0 \
        --metrics $metrics \
        --csv \
        --log-file log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt $tcnt &
        sid=$!
        sleep 20

        cd $pwd; 

        ./run-nlp.sh $task $port $input
        sleep 5
        mv ../dnn/log.csv $stats/prof_${task}_${batch}.csv

        # gpu timing pass
        kill $sid
        cd ../dnn/
        nvprof --devices 0 \
        --print-gpu-summary \
        --csv \
        --log-file log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt $tcnt &
        sid=$!
        sleep 20

        cd $pwd; 

        ./run-nlp.sh $task $port $input
        sleep 5
        mv ../dnn/log.csv $stats/all_${task}_${batch}.csv
        kill $sid
    done
done

cp -rf $stats stats-bk/
