GLOG_logtostderr=1 ./treadmill --config_in_file=workload_config.json \
                               --service_type=dnn \
                               --runtime=20 \
                               --request_per_second=1000 \
                               --port=8080
