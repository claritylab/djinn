req_type=$2
export PROF_REQ_TYPE=$req_type
#nvprof --output-profile ./prof.%q{PROF_REQ_TYPE}.%p.out --devices 0 --log-file ./profiling/prof.%q{PROF_REQ_TYPE}.%p.out ./dnn-server --portno 8398 --debug 1 --gpu $1
./dnn-server --portno 8398 --debug 1 --gpu true
#nvprof  --devices 0 --log-file ./profiling/prof.%q{PROF_REQ_TYPE}.%p.out ./dnn-server --portno 8398 --debug 1 --gpu $1
