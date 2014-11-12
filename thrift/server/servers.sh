numservers=$1
id=$2

mkdir -p log_server
rm -rf `pwd`/log_server/*
pkill DnnServer
for i in $(seq 1 $numservers);
do
  GLOG_stderrthreshold=1 GLOG_log_dir=`pwd`/log_server CUDA_VISIBLE_DEVICES=$id ./DnnServer --gpu=true \
                                                --gpuid=$id \
                                                --num_workers=1 \
                                                --port=$((7999 + $i)) &
done

