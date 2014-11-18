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

sweep = ('1', '16', '32', '128', '256', '512', '1024', '2048')
stats = ('achieved_occupancy', 'sm_efficiency')

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

def util_single ( stat, profl, timl, app ):
    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fig = pl.figure()
    fig.set_size_inches(9, 7)

    col = color_maker(len(profl), map="gnuplot")

    agg = []
    loc = []
    # get average for metric
    for filename in profl:
        batch = get_num(re.findall(r'\d+', filename))
        loc.append(batch)
        occ = []
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if row["Metric Name"] == stat:
                    p = re.search("%", row["Avg_x"])
                    if p:
                        avg = get_num(row["Avg_x"])/100
                    else:
                        avg = get_num(row["Avg_x"])
                    occ.append(float(avg)*float(row["Time(%)"]))
        agg.append(sum(occ))

    w = 0.5
    ind = np.arange(len(loc))+w/2
    ax1 = fig.add_subplot(111, title="")
    ax1.bar(ind, agg, w, color="b",label=stat)
    ax1.set_xticks(ind+w/2)
    ax1.set_xticklabels( sweep )
    ax1.set_ylim([0,100])
    ax1.set_ylabel(stat)

    # get Qps
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

def util_both ( stat, profl, timl, app ):
    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fig = pl.figure()
    fig.set_size_inches(9, 7)

    col = color_maker(len(profl), map="gnuplot")

    metric = {}
    loc = []
    agg = []
    # get average for metric
    for filename in profl:
        batch = get_num(re.findall(r'\d+', filename))
        loc.append(batch)
        for s in stat:
            metric[s] = []
            time = []
            with open(filename, 'rb') as f:
                data = csv.DictReader(f)
                for row in data:
                    if row["Metric Name"] == s:
                        p = re.search("%", row["Avg_x"])
                        if p:
                            avg = get_num(row["Avg_x"])/100
                        else:
                            avg = get_num(row["Avg_x"])
                        metric[s].append(float(avg))
                        time.append(float(row["Time(%)"]))
        v1 = [s1 * s2 for s1,s2 in zip(metric[stat[0]],metric[stat[1]])]
        a = [v * t for v,t in zip(v1,time)]
        agg.append(sum(a))

    w = 0.5
    ind = np.arange(len(loc))+w/2
    ax1 = fig.add_subplot(111, title="")
    ax1.bar(ind, agg, w, color="b",label=stat)
    ax1.set_xticks(ind+w/2)
    ax1.set_xticklabels( sweep )
    ax1.set_ylim([0,100])
    ax1.set_ylabel(stat)

    # get Qps
    ax2 = ax1.twinx()
    count = 0
    qps = []
    loc = []
    for filename in timl:
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                q=float(row["bqpms"])*1000
                qps.append((int(row["batch"]), q))

    qps.sort()
    qs = [ q[1] for q in qps ]
    ax2.scatter(ind+w/2, qs, s=50, c="r", marker='o')
    ax2.set_ylabel('Qps')

    pl.xlabel('Batch Size')
    tit = app + "_agg" 
    pl.title(tit)
    out = tit + ".png"
    pl.savefig(out)

# arg 1 = list of csvs
# arg 2 = app
def main( args ):
    csvs = [ line.strip() for line in open(args[1]) ] # rm /n
    prof = "prof_" + args[2] + "_"
    tim = "timing_" + args[2] + "_"
    dur = "all_" + args[2] + "_"
    profl = []
    timl = []
    durl = []
    for i in csvs:
        p = re.search(prof, i)
        t = re.search(tim, i)
        d = re.search(dur, i)
        if p:
            profl.append(i)
        if t:
            timl.append(i)
        if d:
            durl.append(i)
    
    profl.sort()
    durl.sort()
    timl.sort()
    temp = []
    NUM_FIELDS = 8
    for p,d in zip(profl,durl):
        p_csv = pd.read_csv(p)
        if len(p_csv.columns) > NUM_FIELDS:
            break
        d_csv = pd.read_csv(d)
        p_csv = p_csv.merge(d_csv, left_on='Kernel', right_on='Name', how='outer')
        p_csv.to_csv(p, sep=",")

    # for s in stats:
    #     util_single(s, profl, timl, args[2])

    util_both(stats, profl, timl, args[2])

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
