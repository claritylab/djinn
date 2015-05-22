network=$1
input=$2
trial=$3
gpu=$4

# metrics='achieved_occupancy,ipc'
# metrics='l1_shared_utilization"
metrics='l2_utilization,dram_utilization,eligible_warps_per_cycle'

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
                 --input $input \
                 --trial $trial \
                 --gpu $gpu
