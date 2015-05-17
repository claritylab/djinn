export GLOG_logtostderr=0
export GLOG_log_dir=`pwd`/log
./img-client --hostname localhost \
             --task $1 \
             --portno $2 \
             --input $3 \
             --num $4 \
             --flandmark data/flandmark.dat \
             --haar data/haar.xml \
             --debug 1
