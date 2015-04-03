freq=$1
thread=$2
csv=$3
gpu=$4

#sudo cpufreq-set -f $freq

export OPENBLAS_NUM_THREADS=$thread

./dnn-server \
  --portno 8080 \
  --cputhread $thread \
  --csv $csv \
  --threadcnt 1 \
  --queries 3 \
  --gpu $gpu \
  --cpufreq $freq \
  --debug 0
