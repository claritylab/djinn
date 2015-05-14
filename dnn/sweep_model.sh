#!/bin/bash

include_transfer=false
csv=model-sweep-no-transfer.csv

./local_run.sh imc true $include_transfer  $csv input/imc.in
./local_run.sh dig true $include_transfer  $csv input/dig.in
./local_run.sh face true $include_transfer  $csv input/face.in
./local_run.sh googlenet true $include_transfer  $csv input/googlenet.in

./local_run.sh asr true $include_transfer  $csv input/asr.in

./local_run.sh pos true $include_transfer  $csv input/pos.in
./local_run.sh ner true $include_transfer  $csv input/ner.in
./local_run.sh chk true $include_transfer  $csv input/chk.in
