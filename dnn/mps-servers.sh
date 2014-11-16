numservers=$1
id=$2
BASE_PORT=7999

pkill dnn-server$id
for i in $(seq 1 $numservers);
do
  CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$id ./dnn-server$id --portno $((BASE_PORT + $i)) --gpu 1 &
done
