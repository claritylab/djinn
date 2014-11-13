workers=1
runtime=10
qps=20
log=log_treadmill_${workers}_${qps}_${runtime}.json

GLOG_logtostderr=1 ./treadmill --config_in_file=treadmill_config.json \
                               --service_type=dnn \
                               --number_of_workers=$workers \
                               --runtime=$runtime \
                               --request_per_second=20 \
                               --output_file=$log
                               --port=$1
