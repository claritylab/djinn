export GLOG_logtostderr=1
export GLOG_log_dir=`pwd`/log
./img-client --hostname localhost \
             --task $1 \
             --portno $2 \
             --input input/$1-input.bin \
             --flandmark data/flandmark.dat \
             --haar data/haar.xml \
             --debug 0
