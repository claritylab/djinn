#!/bin/bash

# metrics="achieved_occupancy,sm_efficiency_instance,ipc_instance,l2_utilization,alu_fu_utilization"
metrics="achieved_occupancy,sm_efficiency,ipc"
# metrics="achieved_occupancy,sm_efficiency,ipc"
# metrics="achieved_occupancy,ipc"

gpuid=7
# aggregate stats
agg='on'
# num_servers=( 1 2 4 8 16 )
num_servers=( 2 )

# number of queries to 1 server
NumQueries=1

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
 
for task in imc; do
    stats=$pwd/mps-$task-nvvp
    mkdir -p $stats
    # rm -rf $stats/*
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
##########################################
# GPU Metrics Pass
##########################################
        cd ../dnn/

        ./nvprof-all-nvvp.sh $stats $task $servers
        sleep 1

        for instance in $(seq 1 $servers);
        do
            CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$gpuid ./dnn-server$gpuid --portno $((port + $instance)) \
            --debug 0 \
            --gpu 1 \
            --csv trash.csv \
            --threadcnt $(( $NumQueries*$ThreadsPerQuery )) &
        done
        sleep 20

        cd $pwd;
        ./clients.sh $task $servers $NumQueries $batch $gpuid
        # clean up
        sleep 1
        pkill dnn-server$gpuid
        rm -rf /tmp/.nvprof/nvprof.lock
        pkill nvprof
        echo nvvp done.
    done
done

