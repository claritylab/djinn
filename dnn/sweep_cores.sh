network=$1
freq=$2
thread=$3
csv=$4
gpu=$5

sudo cpufreq-set -f $freq

sudo ./enable_cpu_cores.sh $thread

export OPENBLAS_NUM_THREADS=$thread

./dnn-server \
  --network $network \
  --transfer false \
  --gpu $gpu \
  --cputhread $thread \
  --csv $csv \
  --cpufreq $freq \
  --verbose true \
  --trial 5 \
  --input input/face.in \
  --debug 0
