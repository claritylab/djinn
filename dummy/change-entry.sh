#!/bin/sh

NET=$1
ENTRY=$2
VAL=$3

sed -i -e "0,/${ENTRY}:.*/s/${ENTRY}:.*/${ENTRY}:\ ${VAL}/" $NET
