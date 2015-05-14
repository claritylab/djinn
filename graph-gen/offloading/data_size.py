#!/usr/bin/env python
import pdb
import copy
import sys
import os
import csv
import cv2 
import math 
import numpy as np
import numpy.random as ran
import scipy
import scipy.linalg as linalg
import matplotlib.cm as cmx
import matplotlib.colors as cl
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 

def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                               cmap=plt.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        if height < 0.1:
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'%float(height), ha='center', va='bottom')


def main(argv):
    
    # Plot
    import matplotlib
    # Force matplot lib to not use any Xwindows backend
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    
    plt.rc("font", family="serif")
    plt.rc("font", size=15)
    plt.rc('legend',fontsize=15)
    fsize='xx-small'
    fontP = FontProperties()
    fontP.set_size(fsize)
    fig,ax1 = plt.subplots()
    fig.set_size_inches(15,5)
    width = 0.2
    linewidth = 2

    # Read in the data
    model_list = ['imc', 'face', 'asr', 'pos', 'ner', 'chk']

    colors = color_maker(len(model_list), min=0.1, max=0.9)
    
    max_num_layers = 0

    for model_idx,model in enumerate(model_list):
        # Read in each model's csv
        csv_name = "layers_" + model + ".csv"
        with open(csv_name, 'rb') as f:
            data = csv.DictReader(f)
            layers = []
            data_size = []
            for row in data:
                layers.append(row['layer'])
                data_size.append(float(row['output']) * 4.0 /1024.0) # convert to KB
            
            # Add the beginning data size
            for row in data:
                layers = np.insert(layers, 0, 'BEGIN')
                data_size = np.insert(data_size, 0, float(row['input']) *4.0 / 1024.0)
                break; # only need the input to first layer, break after that

            # plot
            x_values = np.arange(len(layers))

            max_num_layers = max(len(layers), max_num_layers)

            ax1.plot(x_values, data_size, color=colors[model_idx], linewidth=linewidth, label=model)

    ax1.set_xlabel("layers")
    ax1.set_ylabel("Output Data Sizes(KB)")

    x_values = np.arange(max_num_layers)

    ax1.set_xticks(x_values)

    plt.savefig('data_size_pattern.png', bbox_inches='tight')
#    plt.savefig(model+'.eps', bbox_inches='tight')
#    os.popen('epstopdf '+model+'.eps')
    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
