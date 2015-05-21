#!/bin/sh

NET=$1
OCC=$2
VAL=$3

tr "\n" "^" < $NET | sed "s/dim:\ [0-9]*/dim:\ ${VAL}/${OCC}" | tr '^' '\n' > temp.prototxt
mv temp.prototxt $NET
