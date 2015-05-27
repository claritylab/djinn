#!/bin/sh

sed -i -e "0,/input_dim:.*/s/input_dim:.*/input_dim:\ ${2}/" net-configs/$1.prototxt
