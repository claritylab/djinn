# Script for local experiments

network=$1
gpu=$2
include_transfer=$3
csv=$4
input=input/${network}.in
trial=$5

export OPENBLAS_NUM_THREADS=4

./dnn-server --gpu $gpu --transfer $include_transfer --csv $csv --trial $trial --network $network --input $input
