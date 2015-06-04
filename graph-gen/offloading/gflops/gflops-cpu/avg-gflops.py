#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from pprint import pprint # Pretty Print

import pandas as pd
import numpy as np
import math
from pylab import *

import matplotlib.cm as cmx
import matplotlib.colors as cl
import matplotlib.pyplot as pl

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 

def main( args ):
    layers = {}
    names = ['conv', 'local', 'sig','relu','htanh','norm','pool','prob','argmax','drop']
    fc = ['inner', 'ip', 'fc']
    layers['fc'] = []

    # f1='mobile-cpu-layer.csv'
    f1='server-cpu-layer.csv'
    f2='layer-flops.csv'
    # outname="mobile-gflops-cpu.csv"
    outname="server-gflops-cpu.csv"

    # merge flops and timing csvs
    flop_csv = pd.read_csv(f1)
    t_csv = pd.read_csv(f2)
    data = t_csv.merge(flop_csv)

    # calc gflops
    data.lat = data.lat / pow(10,3)
    data['gflops'] = (data.flops / data.lat) / pow(10,9)

    # collect
    for idx,row in data.iterrows():
        for i,n in enumerate(names):
            if n in row['layer']:
                if n not in layers:
                    layers[n] = []
                layers[n].append( row['gflops'] )
        for i,n in enumerate(fc): # treat FC special case
            if n in row['layer']:
                layers['fc'].append( row['gflops'] )
    
    # print layers

    avgs = {}
    std = {}
    for k in layers:
        avgs[k] = np.median(layers[k])
        std[k] = np.std(layers[k])

    w = csv.writer(open(outname, "wb"))
    w.writerow(['layer','gflops'])
    for k,v in avgs.items():
        w.writerow([k,v])

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
