#!/bin/bash

batch_size=$1

input="input/batch-input.txt"

sentence="Rockwell International Corp. 's Tulsa unit said it signed a tentative agreement extending its contract with Boeing Co. to provide structural parts for Boeing 's 747 jetliners . "

echo "" > $input

for i in `seq 1 "$batch_size"`;
do
        sed -i "s|$|$sentence|" $input;
done
echo "Generate batch input : $batch_size done.";
