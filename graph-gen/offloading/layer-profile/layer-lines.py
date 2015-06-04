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

def main( args ):
    apps = ['imc', 'dig', 'face', 'asr', 'pos', 'chk', 'ner']
    apps = ['imc', 'dig-10', 'face']

    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fsize='x-small'
    fontP=FontProperties()
    fontP.set_size(fsize)
    fig = pl.figure()
    fig.set_size_inches(8, 4)

    fname='layer-sizes.csv'
    size=5
    margin = 0.15
    ax1 = fig.add_subplot(111, title="")
    col = color_maker(len(apps), map="gnuplot")

    for idx,app in enumerate(apps):
        l = []
        d = []
        idx = 0
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if app == row['network']:
                    if not l:
                        d.append(float(row['in_size']))
                        l.append("input")
                    d.append(float(row['out_size']))
                    l.append(row['layer'])
        num_items=len(d)
        w = (1.-2.*margin)/num_items
        w = 0.4
        pos = np.arange(num_items)+w
        ax1.plot(pos, d, label=app)

        l1 = ax1.legend(title='application',loc='upper right', prop=fontP, ncol=1)
        l1 = ax1.legend(loc='upper right', prop=fontP, ncol=1)
        # ax1.set_xticks(pos+w/2)
        # ax1.set_xticklabels( l )
        ax1.set_xlabel('Layers')
        ax1.set_ylabel('Size (MB)', rotation=90)
        pl.setp(l1.get_title(), fontsize=fsize)
        # pl.show()
    import os as mars_awesome_os;
    import matplotlib.pyplot as mars_awesome_plt;
    mars_awesome_plt.savefig('layer-lines.eps', bbox_inches='tight');
    mars_awesome_os.popen('epstopdf layer-lines.eps');

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
