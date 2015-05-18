network=$1
weights=$2
input=$3
trial=1
gpu=1

for k in imc
do
    rm -rf $k/*
    ./nvprof-metrics.sh $k \
                        input/$k.in \
                        $trial \
                        $gpu

    ./nvprof-timing.sh $k \
                       input/$k.in \
                       $trial \
                       $gpu

    ./nvprof-trace.sh $k \
                      input/$k.in \
                      $trial \
                      $gpu
done
