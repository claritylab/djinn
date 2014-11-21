#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from pprint import pprint # Pretty Print

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as cl
import matplotlib.pyplot as pl

def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                               cmap=pl.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]

def get_num(x):
    return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))

def main( args ):
    apps = ['imc', 'asr']

    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fsize='x-small'
    fontP=FontProperties()
    fontP.set_size(fsize)
    fig = pl.figure()
    fig.set_size_inches(7, 5)

    filename='timing.csv'
    w=0.2
    size=5
    ax1 = fig.add_subplot(111, title="")
    ax2 = ax1.twinx()
    col = color_maker(len(apps), map="gnuplot")
    for idx,app in enumerate(apps):
        qps = []
        lat = []
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if app == row['app']:
                    lat.append( (int(row["batch"]), float(row["lat"]) ))
                    qps.append( (int(row["batch"]), float(row["qpms"])*1000) )
        qps.sort()
        lat.sort()
        ii = [ i[0] for i in qps ]
        ind = np.arange(len(ii))
        lats = [ l[1] for l in lat ]
        qs = [ q[1] for q in qps ]
        ax1.plot(ind+w/2, lats, '^-', label=app)
        ax2.plot(ind+w/2, qs, 'o-', label=app)

    l1 = ax1.legend(title='latency',loc='upper left', prop=fontP, ncol=1)
    l2 = ax2.legend(title='qps',loc='upper left', prop=fontP, bbox_to_anchor=[0.135, 1], ncol=1)
    ax1.set_ylim([0,200])
    ax1.set_xticks(ind+w/2)
    ax1.set_xticklabels( ii )
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (ms)', rotation=90)
    ax2.set_ylabel('QPS', rotation=-90, labelpad=15)
    pl.setp(l1.get_title(), fontsize=fsize)
    pl.setp(l2.get_title(), fontsize=fsize)

    tit =  "Latency and QPS vs Batch Size"
    pl.title(tit)
    tit = "lat-tp.png"
    pl.savefig(tit)

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
