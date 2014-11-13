#!/bin/bash

# time to sleep between processes
t=10;
stats_chill=30
# dnn server params
for gpu in 1 2 4 8
do
  for rps in 10 25 50
  do
    numservers=$gpu
    serverid=0

    # client params
    numclients=$numservers
    clientid=1

    # runner params
    numrunners=$numclients
    rps=$rps
    # in seconds
    runtime=300

    echo "Setting up server and clients..."
    cd server > /dev/null;
    ./servers.sh $numservers $serverid
    cd - && cd client > /dev/null;
    ./clients.sh $numclients $clientid
    cd - > /dev/null;

    sleep $t;
    echo "Set up done."

    # start collecting stats
    ulog=gpu-stats_util_${numservers}_${numrunners}_${rps}_${runtime}.txt
    nvidia-smi stats -d gpuUtil -i $serverid --filename $ulog &
    pid_stats=$!

    # chill...
    sleep $stats_chill;

    echo "Let's run."
    ./runners.sh $numrunners $rps $runtime

    sleep $runtime;
    kill $pid_stats
  done
done

pkill Dnn*
pkill treadmill
