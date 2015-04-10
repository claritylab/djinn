#!/bin/bash

./local_run.sh imc true model-sweep.csv input/imc.in
./local_run.sh dig true model-sweep.csv input/dig.in
./local_run.sh face true model-sweep.csv input/face.in
./local_run.sh googlenet true model-sweep.csv input/googlenet.in

./local_run.sh asr true model-sweep.csv input/asr.in

./local_run.sh pos true model-sweep.csv input/pos.in
./local_run.sh ner true model-sweep.csv input/ner.in
./local_run.sh chk true model-sweep.csv input/chk.in
