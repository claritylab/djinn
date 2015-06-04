#!/bin/bash

for model in imc face vgg dig asr pos ner chk;do
  for mob_plat in gpu cpu;do
#    for network in lte 3g wifi;do
    for network in lte 3g wifi;do
#      for server_plat in cpu gpu;do
      for server_plat in gpu;do
        ./neurosurgeon.py $model ../layer_profile/layers_${model}.csv \
          $network \
          $mob_plat jetson-${mob_plat}-model.csv ../mobile-${mob_plat}-layer.csv \
          $server_plat server-${server_plat}-model.csv ../server-${server_plat}-layer.csv   
      done
    done
  done
done

