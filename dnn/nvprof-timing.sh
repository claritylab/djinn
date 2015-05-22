network=$1
input=$2
trial=$3
gpu=$4

fname=${network##*/}
export TASK=${fname%%.*}
stats=${fname%%.*}
mkdir -p $stats
rm -rf $stats/summary*

nvprof \
    --aggregate-mode on \
    --print-gpu-summary \
    --csv \
    --log-file $stats/summary_%q{TASK}_%p.csv \
    ./dnn-server --network $network \
                 --input $input \
                 --trial $trial \
                 --gpu $gpu
