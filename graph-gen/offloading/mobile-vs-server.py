#!/usr/bin/env python

import sys
import os # OS stuff
import glob # pathname stuff
import csv # CSV
import re # Regex
from pprint import pprint # Pretty Print

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as cl
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 

import pandas as pd
import numpy as np

fsize='xx-small'
fontP = FontProperties()
fontP.set_size(fsize)
width = 0.2
linewidth = 2

def main():
    # the network to plot
    network = ['imc', 'face', 'VGG', 'dig-10', 'asr', 'pos', 'ner', 'chk']

    server_lat = {}
    server_csv = "server_lats/timing_avg.csv"
    with open(server_csv, 'rb') as f:
        data = csv.DictReader(f)
        for row in data:
            model = row['model']
            plat = row['plat']
            key = (model, plat)
            if key not in server_lat:
                server_lat[key] = float(row['latency'])
            else:
                print "duplicate entries"
                exit(1)

 
    mobile_lat = {}
    mobile_csv = "max_clock/timing_avg.csv"
    with open(mobile_csv, 'rb') as f:
        data = csv.DictReader(f)
        for row in data:
            model = row['model']
            plat = row['plat']
            key = (model, plat)
            if key not in mobile_lat:
                mobile_lat[key] = float(row['latency'])
            else:
                print "duplicate entries"
                exit(1)       

    # both latencies in the dictionary
    # make the array for plotting
    server_cpu_list = []
    server_gpu_list = []
    mobile_cpu_list = []
    mobile_gpu_list = []
    for net in network:
        key = (net, 'cpu')
        server_cpu_list.append(server_lat[key])
        mobile_cpu_list.append(mobile_lat[key])
        key = (net, 'gpu')
        server_gpu_list.append(server_lat[key])
        mobile_gpu_list.append(mobile_lat[key])

    
    # plot
    import matplotlib
    from matplotlib import pyplot as plt
    
    plt.rc("font", family="serif")
    plt.rc("font", size=15)
    plt.rc('legend',fontsize=15)

    fig,ax1 = plt.subplots()
    fig.set_size_inches(10,5)
    barwidth = 0.2
    
    x_values = np.arange(len(network))
    ax1.bar(x_values, server_cpu_list, width=barwidth, label='Server CPU', color='r')
    ax1.bar(x_values+width, server_gpu_list, width=barwidth, label='Server GPU', color='b')
    ax1.bar(x_values+2*width, mobile_cpu_list, width=barwidth, label='Mobile CPU', color='y')
    ax1.bar(x_values+3*width, mobile_gpu_list, width=barwidth, label='Mobile GPU', color='g')

    ax1.set_xlabel('Networks')
    ax1.set_xticks(x_values + 1.5*width)
    ax1.set_xticklabels(network, rotation=30)
    ax1.set_xlim([-1, len(network)+1])

    ax1.legend(ncol=2, prop=fontP)

    plt.savefig("mobile-vs-server.png", bbox_inches='tight')

    return 0

if __name__=='__main__':
    sys.exit(main())
