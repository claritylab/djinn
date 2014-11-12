#!/bin/bash

# time to sleep between processe
t=10;
stats_chill=30
# dnn server params
for i in 2 4 6 8
do
numservers=$i
serverid=0

# client params
numclients=$numservers
clientid=1

# runner params
numrunners=$numclients
rps=25
# in seconds
runtime=120

echo "Setting up server and clients..."
cd server > /dev/null;
./servers.sh $numservers $serverid
cd - && cd client > /dev/null;
./clients.sh $numclients $clientid
cd - > /dev/null;

sleep $t;
echo "Set up done."

# start collecting stats
ulog=gpu-stats_util_${numservers}_${numrunners}_${rps}.txt
nvidia-smi stats -d gpuUtil -i $serverid --filename $ulog &
pid_stats=$!

# chill...
sleep $stats_chill;

echo "Let's run."
./runners.sh $numrunners $rps $runtime

sleep $stats_chill;
kill $pid_stats
done
