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

mps = [1, 2, 4, 8, 16]
stats = ('achieved_occupancy', 'sm_efficiency', 'ipc')

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
    col = color_maker(len(profl), map="gnuplot")
    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fig = pl.figure()
    fig.set_size_inches(9, 5)

    all_server = []
    # get average for metric
    for conf in mps:
        occ = []
        per_server = []
        duration = 0
        for filename in profl:
            servers = int((re.search("\d+",(re.search("_(\d+)_", filename).group(0)))).group(0))
            # go through all server confs csvs
            if servers != conf:
                continue
            with open(filename, 'rb') as f:
                data = csv.DictReader(f)
                for row in data:
                    if row["Metric Name"] == stat:
                        avg = get_num(row["Avg_x"])
                        if re.search("%", row["Avg_x"]): # if value is 0-100
                            avg = get_num(row["Avg_x"])/100
                        occ.append(float(avg)*float(row["Time(%)"]))
                        duration += float(row["Time(%)"])
            per_server.append(sum(occ)/duration)
        all_server.append((conf, sum(per_server)))

    # make separate lists for labels and results
    loc = [ a[0] for a in all_server ]
    agg = [ a[1] for a in all_server ]

    w = 0.5
    ind = np.arange(len(loc))+w/2
    ax1 = fig.add_subplot(111, title="")
    ax1.bar(ind, agg, w, color="b",label=stat)
    ax1.set_xticks(ind+w/2)
    ax1.set_xticklabels( loc )
    if stat == 'ipc':
        ax1.set_ylim([0,7])
    else:
        ax1.set_ylim([0,1])
    ax1.set_ylabel(stat)

    # get Qps
    ax2 = ax1.twinx()
    qps = []
    loc = []
    batch = 0 # get the batch size for title
    for conf in mps:
        s = 0
        num = 0
        with open(timl, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                batch = row["batch"]
                if int(row["servers"]) == conf:
                    s += float(row["qpms"])*1000
                    num += 1
        qps.append((conf, float(s)))

    qps.sort()
    qs = [ q[1] for q in qps ]

    w = 0.5
    ax2.scatter(ind+w/2, qs, s=50, c="r", marker='o')
    ax2.set_yscale('log')
    ax2.set_ylabel('Qps', rotation=-90, labelpad=15)
    ax1.set_xlabel('# DNN Servers')

    tit = app + "_" + batch + "_" + stat
    pl.title(tit)
    out = tit + ".png"
    pl.savefig(out)

def util_both ( stat, profl, timl, app ):
    col = color_maker(len(profl), map="gnuplot")
    pl.rc("font", family="serif")
    pl.rc("font", size=12)
    pl.rc('legend', fontsize=11)
    fig = pl.figure()
    fig.set_size_inches(9, 5)

    metric = {}
    time = {}
    loc = []
    ag = []
    # get average for metric
    for filename in profl:
        batch = get_num(re.findall(r'\d+', filename))
        loc.append(int(batch))
        for s in stat:
            metric[s] = []
            time[s] = []
            dur = 0
            with open(filename, 'rb') as f:
                data = csv.DictReader(f)
                for row in data:
                    if row["Metric Name"] == s:
                        avg = get_num(row["Avg_x"])
                        if re.search("%", row["Avg_x"]):
                            avg = get_num(row["Avg_x"])/100
                        metric[s].append(float(avg))
                        time[s].append(float(row["Time(%)"]))
                        dur += float(row["Time(%)"])
        v1 = [s1 * s2 for s1,s2 in zip(metric[stat[0]], metric[stat[1]])]
        v2 = [(s1 * s2) for s1,s2 in zip(v1, time[stat[0]])]
        ag.append((int(batch), sum(v2)/float(dur)))

    loc.sort()
    ag.sort()
    agg = [ a[1] for a in ag ]
    w = 0.5
    ind = np.arange(len(loc))+w/2
    ax1 = fig.add_subplot(111, title="")
    ax1.bar(ind, agg, w, color="b",label=stat)
    ax1.set_xticks(ind+w/2)
    ax1.set_xticklabels( loc )
    ax1.set_ylim([0,1])
    ax1.set_ylabel(stat)

    # get Qps
    ax2 = ax1.twinx()
    qps = []
    loc = []
    for filename in timl:
        s = 0
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                s += float(row["lat"])
            q = float((int(row["batch"]))/s*1000)
            qps.append((int(row["batch"]), q))

    qps.sort()
    qs = [ q[1] for q in qps ]
    ax2.scatter(ind+w/2, qs, s=50, c="r", marker='o')
    ax2.set_yscale('log')
    ax2.set_ylabel('Qps', rotation=-90, labelpad=15)

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

    # cat all timing together
    t = []
    for p in timl:
        # get num servers from name
        servers = int((re.search("\d+",(re.search("_(\d+)_", p).group(0)))).group(0))
        c = pd.read_csv(p)
        # add column for num servers
        c["servers"] = servers
        t.append(c)
    cat = pd.concat(t, join='inner')
    timl = "all_timing.csv"
    cat.to_csv(timl, sep=",")

    NUM_FIELDS = 8
    for p,d in zip(profl,durl):
        p_csv = pd.read_csv(p)
        if len(p_csv.columns) > NUM_FIELDS:
            continue
        d_csv = pd.read_csv(d)
        p_csv = p_csv.merge(d_csv, left_on='Kernel', right_on='Name', how='outer')
        p_csv.to_csv(p, sep=",")

    for s in stats:
        util_single(s, profl, timl, args[2])

    # util_both(('achieved_occupancy', 'sm_efficiency'), profl, timl, args[2])

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
