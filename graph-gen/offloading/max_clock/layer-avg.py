#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from pprint import pprint # Pretty Print

import pandas as pd
import numpy as np

# arg 1 = list of csvs
def main( args ):
    # collect csvs
    csvs = [ line.strip() for line in open(args[1]) ] # rm /n

    COUNT = 10
    for fname in csvs:
        network = {}
        order = []
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if row['layer'] not in network:
                    network[row['layer']] = float(row['latency'])
                    order.append(row['layer'])
                else:
                    network[row['layer']] += float(row['latency'])
        outname = os.path.splitext(fname)[0] + "_avg.csv"
        writer = csv.writer(open(outname, 'w'))
        writer.writerow(['layer','avg_latency'])
        for l in order:
            writer.writerow( [l, float(network[l]/COUNT)] )

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
