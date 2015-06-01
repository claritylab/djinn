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
    outname = 'layer-flops.csv'
    writer = csv.writer(open(outname, 'w'))
    # writer.writerow(['model','layer','flops'])
    writer.writerow(['model','layer','flops'])

    for fname in csvs:
        d = []
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                d.append( (row['layer'], row['fp_ops']) )
        net = os.path.splitext(fname)[0]
        for s in d:
            writer.writerow( [net, s[0], s[1]] )

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
