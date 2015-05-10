#!/bin/bash

for model in imc face vgg asr pos ner chk;do
  for platform in cpu gpu;do
    for network in LTE 3G;do
      ./layers_breakdown.py $model $platform $network 
    done
  done
done

