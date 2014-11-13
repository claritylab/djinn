#!/bin/bash

runtime=60
workers=1

pkill treadmill
# for qps in {10..100..10}
# for qps in {50..100..10}
# do
  # log=log_treadmill_${workers}_${qps}_${runtime}.json
  log=log_treadmill_${workers}_$1_${runtime}.json
  GLOG_logtostderr=1 ./treadmill --config_in_file=treadmill_config.json \
                                 --service_type=dnn \
                                 --number_of_workers=$workers \
                                 --max_outstanding_requests=1000 \
                                 --runtime=$runtime \
                                 --request_per_second=$1 \
                                 --output_file=$log \
                                 --port=8080
# done
