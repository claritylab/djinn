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

MAXREGS = 65536
MAXBLOCKS = 16
WARPSIZE = 32
NUMWARPS = 64

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

def calc_th(x):
    # x = max(16, x - x%16)
    return float(min(1, float( float(MAXREGS/(x*WARPSIZE))/NUMWARPS )))

def get_reg(filename, name):
    with open(filename, 'rb') as f:
        data = csv.DictReader(f)
        for row in data:
            if name in row['Name']:
                return row['Registers Per Thread']
    
    print "Not found"

def active ( profl, rcsvs, apps ):
    outname = 'active.csv'
    writer = csv.write(open(outname, 'w'))
    writer.writerow(['app','batch','ratio'])

    # get average for metric
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
                    if row["Metric Name"] == 'achieved_occupancy':
                        regs = get_reg(rcsvs[idx], row['Name'])
                        th_occ = calc_th(int(regs))
                        avg = get_num(row["Avg_x"])
                        if re.search("%", row["Avg_x"]): # if value is 0-100
                            avg = get_num(row["Avg_x"])/100
                        occ.append(float(avg/th_occ)*float(row["Time(%)"]))
                        dur += float(row["Time(%)"])
            agg.append((int(batch), sum(occ)/float(dur)))
        agg.sort()
        for rat in agg:
            writer.writerow([app, rat[0], rat[1]])
                
# arg 1 = list of csvs
# arg 2 = app
def main( args ):
    # apps = ['imc', 'asr']
    apps = ['imc']

    # collect csvs
    csvs = [ line.strip() for line in open(args[1]) ] # rm /n
    prof = "prof_"
    dur = "summary_"
    regs = "all_"
    profl = []
    durl = []
    regl = []
    for i in csvs:
        p = re.search(prof, i)
        d = re.search(dur, i)
        r = re.search(regs, i)
        if p:
            profl.append(i)
        elif d:
            durl.append(i)
        elif r:
            regl.append(i)
    
    profl.sort()
    durl.sort()
    regl.sort()
    NUM_FIELDS = 8
    # merge profiling and timing csvs
    for p,d in zip(profl,durl):
        p_csv = pd.read_csv(p)
        if len(p_csv.columns) > NUM_FIELDS:
            continue
        d_csv = pd.read_csv(d)
        p_csv = p_csv.merge(d_csv, left_on='Kernel', right_on='Name', how='outer')
        p_csv.to_csv(p, sep=",")

    active(profl, regl, apps)

    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
