network=$1
weights=$2
input=$3
trial=$4
gpu=$5

metrics='achieved_occupancy,ipc'

fname=${network##*/}
export TASK=${fname%%.*}
stats=${fname%%.*}
mkdir -p $stats
rm -rf $stats/metrics*

nvprof \
    --aggregate-mode on \
    --metrics $metrics \
    --csv \
    --log-file $stats/metrics_%q{TASK}_%p.csv \
    ./dnn-server --network $network \
                 --weights $weights \
                 --input $input \
                 --trial $trial \
                 --gpu $gpu
