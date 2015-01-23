#!/bin/bash

# Read params from text files
file=$1

{
    cd lfw >/dev/null
    class=0
    read # don't read first line
    while read line
    do
        dir=$(echo $line | awk {'print $1'})
        for file in $dir/*.jpg
        do
            echo "$file $class"
        done
        let class++
    done
} < $file
