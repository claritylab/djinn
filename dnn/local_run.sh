# Script for local experiments

network=$1
weights=$2
input=$3
trial=$4
gpu=$5

./dnn-server --network $network \
             --weights $weights \
             --input $input \
             --trial $trial \
             --gpu $gpu
