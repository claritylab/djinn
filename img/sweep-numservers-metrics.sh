#!/bin/bash

# metrics="achieved_occupancy,sm_efficiency_instance,ipc_instance,l2_utilization,alu_fu_utilization"
metrics="achieved_occupancy,sm_efficiency,ipc,l2_utilization,l1_shared_utilization"
# metrics="achieved_occupancy,sm_efficiency,ipc"
# metrics="achieved_occupancy,ipc"

gpuid=7
# aggregate stats
agg='on'
num_servers=( 1 2 4 8 16 )

# number of queries to 1 server
NumQueries=16

# nothing to change under here
# number of threads for 1 request
ThreadsPerQuery=1
export GLOG_logtostderr=0
pwd=$PWD;
port=$(( 7999 + $gpuid*100 ))

cd ../dnn/
./extra-scripts/clean-mps.sh
./make-mps.sh $gpuid
cd $pwd;
 
for task in imc dig face; do
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

    for servers in "${num_servers[@]}";
    do
        echo Task: $task FWD passes: $NumQueries servers: $servers
##########################################
# QPS Timing pass
##########################################
        cd ../dnn/
        for instance in $(seq 1 $servers);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $instance)) \
            --debug 0 \
            --gpu 1 \
            --csv $stats/timing_${task}_${servers}_${NumQueries}_${instance}.csv \
            --threadcnt $ThreadsPerQuery \
            --queries $NumQueries &
        done
        sleep 20

        cd $pwd;

        ./clients.sh $task $servers $batch $gpuid
        # clean up
        pkill dnn-server$gpuid

##########################################
# GPU Metrics Pass
##########################################
        cd ../dnn/

        ./nvprof-all-metrics.sh $metrics $stats $task $servers $agg
        sleep 1

        for instance in $(seq 1 $servers);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $instance)) \
            --debug 0 \
            --gpu 1 \
            --csv trash.csv \
            --threadcnt $ThreadsPerQuery \
            --queries $NumQueries &
        done
        sleep 20

        cd $pwd;
        ./clients.sh $task $servers $batch $gpuid
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

        ./nvprof-all-timing.sh $stats $task $servers $agg
        sleep 1

        for instance in $(seq 1 $servers);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $instance)) \
            --debug 0 \
            --gpu 1 \
            --csv trash.csv \
            --threadcnt $ThreadsPerQuery \
            --queries $NumQueries &
        done
        sleep 20

        cd $pwd; 
        ./clients.sh $task $servers $batch $gpuid
        # clean up
        sleep 1
        pkill dnn-server$gpuid
        rm -rf /tmp/.nvprof/nvprof.lock
        pkill nvprof
        echo timing clients done.

##########################################
# GPU Timing pass
##########################################
        cd ../dnn/

        ./nvprof-all-trace.sh $stats $task $servers $agg
        sleep 1

        for instance in $(seq 1 $servers);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $instance)) \
            --debug 0 \
            --gpu 1 \
            --csv trash.csv \
            --threadcnt $ThreadsPerQuery \
            --queries $NumQueries &
        done
        sleep 20

        cd $pwd; 
        ./clients.sh $task $servers $batch $gpuid
        # clean up
        sleep 1
        pkill dnn-server$gpuid
        rm -rf /tmp/.nvprof/nvprof.lock
        pkill nvprof
        echo timing clients done.
    done
done

