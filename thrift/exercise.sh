GLOG_logtostderr=1 ./treadmill --config_in_file=treadmill_config.json \
                               --service_type=dnn \
                               --runtime=10 \
                               --request_per_second=20 \
                               --port=$1
