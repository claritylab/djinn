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

# for each app look for HtoD data size
APPS = {}
APPS['imc'] = 0.618348 # MB
APPS['dig'] = 0
APPS['face'] = 0.277248
APPS['asr'] = 0
APPS['pos'] = 0
APPS['chk'] = 0
APPS['ner'] = 0
APPS['srl'] = 0.112000

SKIPS = 1
STREAMS = 33
THREADS = STREAMS * 2048

pl.rc("font", family="serif")
pl.rc("font", size=12)
pl.rc('legend', fontsize=11)
fig = pl.gcf()
ax = pl.subplot(111)
fig.set_size_inches(9, 7)

# Each function declares at the top its intended accumulators and returned data
# map is the name of one of the colormaps from
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html

def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                               cmap=pl.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]

def util ( filename, app ):
    res = {}
    busy = 0
    t_start = 0
    t_end = 0
    with open(filename, 'rb') as f:
        data = csv.DictReader(f)
        skips = SKIPS
        for row in data:
            if skips > 0 and row['Name'] == '[CUDA memcpy HtoD]' and float(row['Size']) == APPS[app]:
                skips = skips - 1
                continue
            elif skips == 0 and row['Name'] == '[CUDA memcpy HtoD]' and float(row['Size']) == APPS[app]:
                t_start = float(row['Start'])
            elif skips == 0 and row['Grid X'] and t_start != 0:
                grid = int(row['Grid X']) * int(row['Grid Y']) * int(row['Grid Z'])
                block = int(row['Block X']) * int(row['Block Y']) * int(row['Block Z'])
                ths = grid * block
                busy = busy + (ths * float(row['Duration']))/THREADS
            elif skips == 0 and row['Name'] == '[CUDA memcpy DtoH]' and t_start != 0:
                t_end = float(row['Start'])
                break; # break when kernel ends

    print t_end - t_start
    # collect
    res['busy'] = busy
    res['t_start'] = t_start
    res['t_end'] = t_end

    return res

def pcie ( filename, app ):
    total_size = 0
    total_time = 0
    res = {}
    with open(filename, 'rb') as f:
        data = csv.DictReader(f)
        skips = SKIPS
        for row in data:
            if skips > 0 and row['Name'] == '[CUDA memcpy HtoD]' and float(row['Size']) == APPS[app]:
                skips = skips - 1
                continue
            elif skips == 0 and row['Name'] == '[CUDA memcpy HtoD]' and float(row['Size']) == APPS[app]:
                total_time = total_time + float(row['Duration'])
                total_size = total_size + float(row['Size'])
            elif skips == 0 and row['Name'] == '[CUDA memcpy HtoD]' and total_time != 0:
                total_time = total_time + float(row['Duration'])
                total_size = total_size + float(row['Size'])
            elif skips == 0 and row['Name'] == '[CUDA memcpy DtoH]' and total_time != 0:
                total_time = total_time + float(row['Duration'])
                total_size = total_size + float(row['Size'])
                break
                
    output = filename + " pcie time: " + str(total_time) + " ms size: " + str(total_size) + " MB"
    print output
    res['time'] = total_time
    res['size'] = total_size
    return res

# timestamp activity of each stream
def stream ( filename, app ):
    sms = {}
    for s in range(0, STREAMS):
        sms[s] = []
        with open(filename, 'rb') as f:
            data = csv.DictReader(f)
            skips = SKIPS
            for row in data:
                if skips > 0 and row['Name'] == '[CUDA memcpy HtoD]' and float(row['Size']) == APPS[app]:
                    skips = skips - 1
                    continue
                elif skips == 0 and row['Grid X'] and s == int(row['Stream']):
                    sms[s].append(float(row['Start']))
                elif skips == 0 and row['Name'] == '[CUDA memcpy DtoH]':
                    break;

    return sms

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
