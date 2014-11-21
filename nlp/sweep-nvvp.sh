#!/bin/bash
input="input/batch-input.txt"
touch $input
sentence="Rockwell International Corp. 's Tulsa unit said it signed a tentative agreement extending its contract with Boeing Co. to provide structural parts for Boeing 's 747 jetliners . "

batch_size=( 1 16)
pwd=$PWD;

gpuid=2
export CUDA_VISIBLE_DEVICES=$gpuid
port=$(( 7999 + $gpuid*100 ))
 
for task in pos ner chk; do
    stats=$pwd/stats-nvvp-$task
    mkdir -p $stats
    rm -rf $stats/*
    tcnt=1
    for batch in "${batch_size[@]}";
    do
        echo $task $batch
        rm -rf $input
        for i in `seq 1 $batch`;
        do
            echo -n "$sentence" >> $input;
        done
        if [ "$task" == "chk" ]; then
            tcnt=2
        elif [ "$task" == "srl" ]; then
            tcnt=4
        fi
        cd ../dnn/
        nvprof --devices 0 \
        --analysis-metrics \
        --output-profile $stats/log.csv \
        ./dnn-server --portno $port --debug 0 --gpu 1 --csv trash.csv --threadcnt $tcnt &
        sid=$!
        sleep 20

        cd $pwd; 

        ./run-nlp.sh $task $port $input
        mv $stats/log.csv $stats/all_${task}_${batch}.csv
        kill $sid
    done
done



