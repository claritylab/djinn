#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import argparse # Parse inputs
import re # Regex
from pprint import pprint # Pretty Print

import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as cl
import matplotlib.pyplot as pl

pl.rc("font", family="serif")
pl.rc("font", size=12)
pl.rc('legend', fontsize=11)
fig = pl.gcf()
ax = pl.subplot(111)
fig.set_size_inches(9, 7)

def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                               cmap=pl.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]

def pcie ( filename, app ):
    total_size = 0
    total_time = 0
    res = {}
    with open(filename, 'rb') as f:
        data = csv.DictReader(f)
        skips = SKIPS
        for row in data:

def plt_sm ( csvs, app ):
    color = color_maker(len(csvs), map="gnuplot")
    count = 0
    y_axis = []
    leg = []
    for c in csvs:
        data = stream(c, app)
        name = 'Caffe_' + str(count)
        leg.append(name)
        for sm in data:
            if data[sm]:
                y_axis.append(sm)
                y = [sm for i in range(0, len(data[sm]))]
                x = [data[sm][i] - min(data[sm]) for i in range(0, len(data[sm]))]
                pl.scatter(data[sm], y, s=110, c=color[count], marker='|', label=name)
        count = count + 1

    pl.legend(leg)
    pl.yticks(y_axis)
    pl.ylabel('Stream Number')
    pl.xlabel('Execution time')
    out = 'sms-' + app + '.png'
    pl.savefig(out)

def main( args ):
    # arg 1 = list of csvs
    # arg 2 = list of app
    csvs = [ line.strip() for line in open(args[1]) ] # rm /n

    # for c in csvs:
        # util(c, args[2])
        # pcie(c, args[2])

    # hold returned data
    plt_sm( csvs, args[2] )

# needed eventualy
    # print to csvs
    # with open('sm_activity.csv', 'wt') as fp:
    #     a = csv.writer(fp, delimiter=',')
    #     a.writerow( ('sm','time') )

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
