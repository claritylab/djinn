# Script for local experiments

network=$1
gpu=$2
include_transfer=$3
csv=$4
input=$5
trial=$6

./dnn-server --gpu $gpu --transfer $include_transfer --csv $csv --trial $trial --network $network --input $input
