metrics=$1
stats=$2
export TASK=$3
export SERVERS=$4
agg=$5

nvprof --profile-all-processes \
--aggregate-mode on \
--metrics $metrics \
--csv \
--log-file $stats/prof_%q{TASK}_%q{SERVERS}_%p.csv &
