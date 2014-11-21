stats=$1
export TASK=$2
export SERVERS=$3
agg=$4

nvprof --profile-all-processes \
--aggregate-mode on \
--print-gpu-summary \
--csv \
--log-file $stats/all_%q{TASK}_%q{SERVERS}_%p.csv &
