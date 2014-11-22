stats=$1
export TASK=$2
export SERVERS=$3

echo $TASK
echo $SERVERS
nvprof --profile-all-processes -o $stats/nvvp_%p.csv &

# nvprof --profile-all-processes \
# --analysis-metrics \
# --output-profile $stats/nvvp_%p.csv &
