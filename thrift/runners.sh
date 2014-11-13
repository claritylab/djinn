#!/bin/bash

numrunners=$1
rps=$2
runtime=$3
workers=1

pkill treadmill
log=log_treadmill_${workers}_${qps}_${runtime}.json
for i in $(seq 1 $numrunners);
do
  GLOG_logtostderr=1 ./treadmill --config_in_file=treadmill_config.json \
                                 --service_type=dnn \
                                 --number_of_workers=$workers \
                                 --runtime=$runtime \
                                 --request_per_second=$rps \
                                 --output_file=$log \
                                 --port=$((8079 + $i)) &
done
