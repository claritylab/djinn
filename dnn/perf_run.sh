# Script to collect number of fp ops from perf

network=$1
gpu=$2
include_transfer=$3
csv=$4
input=$5
trial=$6

perf stat -e r530410 -e r530110 -e r530810 -B \
  ./dnn-server --gpu $gpu --transfer $include_transfer --csv $csv --trial $trial --network $network --input $input
