numservers=$1
id=$2

mkdir -p log_client
rm -rf `pwd`/log_client/*
pkill DnnClient
for i in $(seq 1 $numservers);
do
  GLOG_stderrthreshold=1 GLOG_log_dir=`pwd`/log_client CUDA_VISIBLE_DEVICES=$id ./DnnClient --port=$((8079 + $i))  \
                                                                                --dnn_port=$((7999 + $i)) &
done
