numservers=$1
id=$2
BASE_PORT=7999

pkill dnn-server
for i in $(seq 1 $numservers);
do
  export CUDA_VISIBLE_DEVICES=$numservers 
  nvprof --print-gpu-trace --csv --log-file $i.csv \
                                    ./dnn-server --portno $((BASE_PORT + $i)) --gpu 1 &
./dnn-server --portno $i --threadno $2 --gpu 1 --debug 1&
done
