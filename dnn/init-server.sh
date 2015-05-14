gpu=$1
csv=$2
threadcnt=$3
queries=$4

./dnn-server \
  --server true \
  --gpu $gpu \
  --portno 8080 \
  --csv $csv \
  --threadcnt $threadcnt \
  --trial $queries \
  --debug 0
