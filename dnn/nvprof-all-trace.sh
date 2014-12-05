stats=$1
export TASK=$2
export SERVERS=$3
agg=$4

nvprof --profile-all-processes \
--aggregate-mode on \
--system-profiling-on \
--print-gpu-trace \
--csv \
--log-file $stats/trace_%q{TASK}_%q{SERVERS}_%p.csv &
