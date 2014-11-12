#!/bin/bash

numrunners=$1
rps=$2
runtime=$3

pkill treadmill
for i in $(seq 1 $numrunners);
do
  GLOG_logtostderr=1 ./treadmill --config_in_file=treadmill_config.json \
                                 --service_type=dnn \
                                 --request_per_second=$rps \
                                 --runtime=$runtime \
                                 --port=$((8079 + $i)) &
done
