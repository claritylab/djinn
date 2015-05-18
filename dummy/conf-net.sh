#!/bin/sh

GPU=1
NET=fc
CONF=$NET.prototxt
OUTNAME=$NET.csv
OUTSIZE=$NET-conf.csv

# rm -rf $OUTNAME $OUTSIZE

# DIM
# FIX OUTPUT AT
dim=10000
# sed -i -e "0,/dim:.*/s/dim:.*/dim:\ ${dim}/" $CONF
for out in 1 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
    # Change output dimension
    sed -i -e "0,/num_output:.*/s/num_output:.*/num_output:\ ${out}/" $CONF

    # Run the network
    ./dummy --gpu $GPU --network $CONF --layer_csv $OUTNAME
    # Append to the csv
    echo "$NET,$dim,$out" >> $OUTSIZE
    sleep 1
done

# OUTPUT
# FIX DIM AT
# DIM=1000
# sed -i -e "0,/dim:.*/s/dim:.*/dim:\ ${DIM}/" $CONF
# for d in 10 100 1000 5000 10000 20000 30000 50000 100000;
# do
#     # Change dimensions
#     sed -i -e "0,/num_output:.*/s/num_output:.*/num_output:\ ${d}/" $CONF
#
#     # Run the network
#     ./dummy --gpu $GPU --network $CONF --layer_csv $OUTNAME
#
#     # Append to the csv
#     echo "$NET,out,$d" >> $OUTSIZE
# done
#
# for d in 10 100 1000 5000 10000 20000;
# do
#     # Change dimensions
#     sed -i -e "0,/dim:.*/s/dim:.*/dim:\ ${d}/" $CONF
#     sed -i -e "0,/num_output:.*/s/num_output:.*/num_output:\ ${d}/" $CONF
#
#     # Run the network
#     ./dummy --gpu $GPU --network $CONF --layer_csv $OUTNAME
#
#     # Append to the csv
#     echo "$NET,both,$d" >> $OUTSIZE
# done

sed -i -e "s/data.*//" $OUTNAME
sed -i -e "/^$/d" $OUTNAME
cat $OUTNAME
cat $OUTSIZE
