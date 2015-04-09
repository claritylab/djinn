# Script for local experiments

network=$1
gpu=$2
csv=$3
input=$4

./dnn-server --gpu $gpu --csv $csv --trial 5 --network $network --input $input
