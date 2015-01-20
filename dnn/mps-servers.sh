numservers=$1
id=$2
tcnt=$3
task=$4
stats=$5

BASE_PORT=$(( 7999 + $id*100 ))

pkill dnn-server$id
for i in $(seq 1 $numservers);
do
  CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$id ./dnn-server$id --portno $((BASE_PORT + $i)) \
                                                       --debug 0 \
                                                       --gpu 1 \
                                                       --csv $stats/timing_${task}_${i}.csv \
                                                       --threadcnt $tcnt &
done
