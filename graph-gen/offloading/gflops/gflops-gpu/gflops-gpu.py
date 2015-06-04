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

def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                               cmap=pl.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]

def plot_gflops( data ):
    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fsize='x-small'
    fontP=FontProperties()
    fontP.set_size(fsize)
    fig = pl.figure()
    fig.set_size_inches(8, 4)

    size=5
    margin = 0.1
    ax1 = fig.add_subplot(111, title="")
    col = color_maker(len(data), map="gnuplot")

    # types = []
    # for k,v in data:
    #     types

    num_items=len(data)
    sep = 0.25
    pos = np.arange(num_items)+sep
    # print data
    # for k in data:
    #     print k
    #     print data[k]

    # d = [data['conv'], data['fc'], data['activation'], data['other']]
    # d = [data['fc'], data['activation'], data['other']]
    # d = [data['other']]

    ax1.plot(*zip(*data['conv']), marker='x', ls='')
    # ax1.plot(*zip(*data['fc']), marker='x', ls='')
    # ax1.plot(*zip(*data['activation']), marker='x', ls='')
    # ax1.plot(*zip(*data['other']), marker='x', ls='')
    ax1.set_xlim([0, 5])

    # l1 = ax1.legend(title='application',loc='upper right', prop=fontP, ncol=1)
    # l1 = ax1.legend(loc='upper right', prop=fontP, ncol=1)
    # ax1.set_xticks(pos+w/2)
    # ax1.set_xticklabels( l )
    # ax1.set_xlabel('Layers')
    # ax1.set_ylabel('Size (MB)', rotation=90)
    # pl.setp(l1.get_title(), fontsize=fsize)

def main( args ):
    layers = {}
    layers['conv'] = []
    conv = ['conv', 'local']
    layers['fc'] = []
    fc = ['inner', 'ip', 'fc']
    layers['activation'] = []
    activations = ['sig','relu', 'htanh']
    layers['other'] = []
    other = ['norm', 'pool', 'prob', 'argmax', 'drop']

    f1='mobile-gpu-layer.csv'
    f2='layer-flops.csv'

    # merge flops and timing csvs
    flop_csv = pd.read_csv(f1)
    t_csv = pd.read_csv(f2)
    data = t_csv.merge(flop_csv)

    # calc gflops
    data.lat = data.lat / pow(10,3)
    data['gflops'] = (data.flops / data.lat) / pow(10,9)
    for idx,row in data.iterrows():
        for c in conv:
            if c in row['layer'] :
                layers['conv'].append( (1, row['gflops']) )
                continue
        for a in activations:
            if a in row['layer'] :
                layers['activation'].append( (2, row['gflops']) )
                continue
        for f in fc:
            if f in row['layer'] :
                layers['fc'].append( (3, row['gflops']) )
                continue
        for o in other:
            if o in row['layer'] :
                layers['other'].append( (4, row['gflops']) )
                continue

    # print layers
    # cover all the layers
    assert(len(layers['fc']) + len(layers['conv']) + len(layers['activation']) + len(layers['other']) == len(data))

    plot_gflops(layers)

    import os as mars_awesome_os;
    import matplotlib.pyplot as mars_awesome_plt;
    mars_awesome_plt.savefig('gflops-gpu.eps', bbox_inches='tight');
    mars_awesome_os.popen('epstopdf gflops-gpu.eps');

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
