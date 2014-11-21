#!/bin/bash

# metrics="achieved_occupancy,sm_efficiency_instance,ipc_instance,l2_utilization,alu_fu_utilization"
metrics="achieved_occupancy,sm_efficiency,ipc"
# metrics="achieved_occupancy,sm_efficiency,ipc,l2_utilization,alu_fu_utilization"

agg='on'
gpuid=7
num_servers=( 1 2 4 8 16 )
# num_servers=( 16 )
tcnt=1
queries=1

# nothing to change under here
export GLOG_logtostderr=0
pwd=$PWD;
port=$(( 7999 + $gpuid*100 ))

cd ../dnn/
./extra-scripts/clean-mps.sh
./make-mps.sh $gpuid
cd $pwd;
 
for task in face; do
    stats=$pwd/mps-$task-$agg
    mkdir -p $stats
    rm -rf $stats/*
    if [ "$task" == "imc" ]; then
        batch=16
    elif [ "$task" == "dig" ]; then
        batch=2
    elif [ "$task" == "face" ]; then
        cd ../dnn
        batch=2
        ./change_batch.sh face $batch
        cd $pwd
    fi

    for s in "${num_servers[@]}";
    do
        echo Task: $task servers: $s
##########################################
# QPS Timing pass
##########################################
        cd ../dnn/
        for i in $(seq 1 $s);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $i)) \
            --debug 0 \
            --gpu 1 \
            --csv $stats/timing_${task}_${s}_${i}.csv \
            --threadcnt $tcnt &
        done
        sleep 20

        cd $pwd;

        ./clients.sh $task $s $batch $gpuid
        # clean up
        pkill dnn-server$gpuid

##########################################
# GPU Metrics Pass
##########################################
        cd ../dnn/

        ./nvprof-all-metrics.sh $metrics $stats $task $s $agg
        sleep 1

        for i in $(seq 1 $s);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $i)) \
            --debug 0 \
            --gpu 1 \
            --csv trash.csv \
            --threadcnt $tcnt &
        done
        sleep 20

        cd $pwd;
        ./clients.sh $task $s $batch $gpuid
        # clean up
        sleep 1
        pkill dnn-server$gpuid
        rm -rf /tmp/.nvprof/nvprof.lock
        pkill nvprof
        echo metrics clients done.

##########################################
# GPU Timing pass
##########################################
        cd ../dnn/

        ./nvprof-all-timing.sh $stats $task $s $agg
        sleep 1

        for i in $(seq 1 $s);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $i)) \
            --debug 0 \
            --gpu 1 \
            --csv trash.csv \
            --threadcnt $tcnt &
        done
        sleep 20

        cd $pwd; 
        ./clients.sh $task $s $queries $gpuid
        # clean up
        sleep 1
        pkill dnn-server$gpuid
        rm -rf /tmp/.nvprof/nvprof.lock
        pkill nvprof
        echo timing clients done.
    done
done

