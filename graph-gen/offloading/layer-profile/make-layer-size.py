#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from pprint import pprint # Pretty Print

import pandas as pd
import numpy as np

import matplotlib.cm as cmx
import matplotlib.colors as cl
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 

BYTES = 4
SIZE = 'MB'

def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                               cmap=plt.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]

def get_kb(num):
    return (float)((float(num)*BYTES)/1024) # KB

def get_mb(num):
    return (float)((float(num)*BYTES)/(1024*1024)) # MB

# arg 1 = list of csvs
def main( args ):
    # collect csvs
    csvs = [ line.strip() for line in open(args[1]) ] # rm /n
    outname = 'layer-sizes.csv'
    writer = csv.writer(open(outname, 'w'))
    writer.writerow(['network','layer','in_size','out_size'])

    for fname in csvs:
        d = []
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if SIZE == 'MB':
                    d.append( (row['layer'], get_mb(row['input']), get_mb(row['output'])) )
                else:
                    d.append( (row['layer'], get_kb(row['input']), get_kb(row['output'])) )
        net = os.path.splitext(fname)[0]
        for s in d:
            writer.writerow( [net, s[0], s[1], s[2]] )

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
