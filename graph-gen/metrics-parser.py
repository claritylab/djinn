#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from pprint import pprint # Pretty Print

import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as cl
import matplotlib.pyplot as pl

# sweep = ('1', '16', '32', '64', '128', '256', '512', '1024', '2048')
sweep = ('1', '16', '32', '128', '256', '512', '1024', '2048')

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

def util_plt ( stat, profl, timl, app ):
    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fig = pl.figure()
    fig.set_size_inches(9, 7)

    col = color_maker(len(profl), map="gnuplot")
    basedir = 'stats/'

    metric = 0
    weight = 0
    avg = 0
    avg_occ = []
    loc = []
    for filename in profl:
        batch = get_num(re.findall(r'\d+', filename))
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if row["Metric Name"] == stat:
                    p = re.search("%", row["Avg"])
                    if p:
                        avg = get_num(row["Avg"])/100
                    else:
                        avg = get_num(row["Avg"])
                    metric = metric + float(avg)
                    weight = weight + 1
            val = float(metric/weight)
            avg_occ.append(val)
            loc.append(batch)

    w = 0.5
    ind = np.arange(len(loc))+w/2
    ax1 = fig.add_subplot(111, title="")
    ax1.bar(ind, avg_occ, w, color="b",label=stat)
    ax1.set_xticks(ind+w/2)
    ax1.set_xticklabels( sweep )
    ax1.set_ylim([0,1])
    ax1.set_ylabel(stat)

    ax2 = ax1.twinx()
    count = 0
    qps = []
    loc = []
    for filename in timl:
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                q=float(row["qpms"])*1000
                qps.append((int(row["batch"]), q))

    qps.sort()
    qs = [ q[1] for q in qps ]
    ax2.scatter(ind+w/2, qs, s=50, c="r", marker='o')
    ax2.set_ylabel('Qps')

    pl.xlabel('Batch Size')
    tit = app + "_" + stat
    pl.title(tit)
    out = tit + ".png"
    pl.savefig(out)

def main( args ):
    # arg 1 = list of csvs
    # arg 2 = app
    csvs = [ line.strip() for line in open(args[1]) ] # rm /n
    basedir = "stats/"
    prof = "prof_" + args[2] + "_"
    tim = "timing_" + args[2] + "_"
    profl = []
    timl = []
    for i in csvs:
        p = re.search(prof, i)
        t = re.search(tim, i)
        if p:
            profl.append(i)
        if t:
            timl.append(i)
    
    stat = "achieved_occupancy"
    util_plt(stat, profl, timl, args[2])
    stat = "sm_efficiency"
    util_plt(stat, profl, timl, args[2])
    stat = "warp_execution_efficiency"
    util_plt(stat, profl, timl, args[2])

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
