network=$1
weights=$2
input=$3
trial=$4
gpu=$5

fname=${network##*/}
export TASK=${fname%%.*}
stats=${fname%%.*}
mkdir -p $stats
rm -rf $stats/trace*

nvprof \
    --aggregate-mode on \
    --print-gpu-trace \
    --csv \
    --log-file $stats/trace_%q{TASK}_%p.csv \
    ./dnn-server --network $network \
                 --weights $weights \
                 --input $input \
                 --trial $trial \
                 --gpu $gpu
