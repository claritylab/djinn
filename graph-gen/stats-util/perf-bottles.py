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

stats = ('ipc', 'l2_utilization', 'l1_shared_utilization')
# stats = ('achieved_occupancy')

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

def util_single ( stat, profl, timl, apps ):
    out = stat + ".csv"
    writer = csv.writer(open(out, 'w'))
    writer.writerow(['app','stat','batch','val'])

    ag = []
    loc = []

    for app in apps:
        agg = []
        for idx, filename in enumerate(profl):
            if not re.search(app, filename): 
                continue
            batch = get_num(re.findall(r'\d+', filename))
            dur = 0
            occ = []
            with open(filename, 'rb') as f:
                data = csv.DictReader(f)
                for row in data:
                    if row["Metric Name"] == stat:
                        avg = get_num(row["Avg_x"])
                        if re.search("%", row["Avg_x"]): # if value is 0-100
                            avg = get_num(row["Avg_x"])/100
                        occ.append(float(avg)*float(row["Time(%)"]))
                        dur += float(row["Time(%)"])
            agg.append((int(batch), sum(occ)/float(dur)))
        agg.sort()
        for val in agg:
            writer.writerow([app, stat, val[0], val[1]])

# arg 1 = list of csvs
# arg 2 = app
def main( args ):
    # apps = ['imc', 'dig', 'face', 'asr']
    apps = ['imc']

    csvs = [ line.strip() for line in open(args[1]) ] # rm /n
    prof = "prof_"
    tim = "timing_"
    dur = "all_"
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
            continue
        d_csv = pd.read_csv(d)
        p_csv = p_csv.merge(d_csv, left_on='Kernel', right_on='Name', how='outer')
        p_csv.to_csv(p, sep=",")

    for s in stats:
        util_single(s, profl, timl, apps)

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
