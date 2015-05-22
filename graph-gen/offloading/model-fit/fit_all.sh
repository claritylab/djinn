#!/bin/bash

for plat in cpu gpu;do
    for c in server jetson;do
        model_csv=${c}-${plat}-model.csv
        echo "layer,function,equation,parameters,p0,p1,p2,p3,p4,stderr" > $model_csv

        # for model in conv fc sig htanh relu pool norm local softmax argmax;do
        for model in conv sig htanh relu pool norm local softmax argmax;do
            ./fit_${model}.py ./${c}/${model}-${plat}-gflops.csv >> $model_csv
        done
    done
done
