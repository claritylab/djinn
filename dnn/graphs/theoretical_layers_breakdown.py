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
import matplotlib.cm as cm

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 
server_tp = {'imc':352.45, 'face':49.50, 'vgg':1008.22, 'asr':3.55, 'pos':1.48, 'ner':1.57, 'chk':1.47} 
mobile_gpu_tp = {'imc':23.15, 'face':5.34, 'vgg':49.42, 'asr':0.18, 'pos':0.32, 'ner':0.34, 'chk':0.29} 
mobile_cpu_tp = {'imc':5.88, 'face':1.36, 'vgg':4.00, 'asr':0.017, 'pos':0.22, 'ner':0.29, 'chk':0.22} 
mobile_cpu_power = {'imc':6.4, 'face':6.4, 'vgg':9.2, 'asr':8.0, 'pos':5.9, 'ner':7.1, 'chk':6.3} 
mobile_gpu_power = {'imc':8.7, 'face':9.5, 'vgg':8.7, 'asr':12.4, 'pos':5.2, 'ner':5.1, 'chk':5.1} 

#########LTE#######
uplink_bw = 5.5 #LTE Mbps
downlink_bw = 24.3 #LTE Mbps
uplink_power = (438.39 * uplink_bw + 1288.04) / 1000.0 # LTE W
downlink_power = (51.97 * downlink_bw + 1288.04) / 1000.0 # LTE W

#######3g###########
#uplink_bw = 1.44
#downlink_bw = 3.84
#uplink_power = (868.98 * uplink_bw + 817.88) / 1000.0 # 3g W
#downlink_power = (122.12 * downlink_bw + 817.88) / 1000.0 # 3g W

def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        if height < 0.1:
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'%float(height), ha='center', va='bottom')


def main(argv):

    # Read in the data
    data_file = str(argv[1]) 

    model = data_file.split('.')[0].split('_')[1]
    reader = csv.reader(open(data_file, 'rb'), delimiter=',')
    mat = np.array(list(reader))

    # Remove the first row
    mat = np.delete(mat, 0, 0)

    # Get the layers
    layers = mat[:,0]

    # Get the computation
    comp = mat[:,5]

    comp = comp.astype(float) / float(math.pow(10,9))
    # Get the data to be offload
    offload = mat[:,2]

    offload = offload.astype(float)

    offload = offload * 32.0 / 1024.0 / 1024.0 #covnert to Mb
    # Get the depth of network
    num_layers = len(offload)
    cpu_latency_list = [] 
    cpu_energy_list = []
    gpu_latency_list = []
    gpu_energy_list = []


    ########################real meausred latency for each layer##########

    # if offload at the very beginning
    data_before_first_layer = mat[0][1]

    server_comp = 0.0
    for i in np.arange(num_layers):
        server_comp += comp[i]

    latency = float(data_before_first_layer) * 32.0 / 1024.0 / 1024.0 / uplink_bw + server_comp / server_tp[model] 

    energy = float(data_before_first_layer) * 32.0 / 1024.0 / 1024.0 / uplink_bw * uplink_power

    cpu_latency_list.append(latency)
    gpu_latency_list.append(latency)
    cpu_energy_list.append(energy)
    gpu_energy_list.append(energy)

    # Sweep through the network to find the sweep spot
    for layer_idx in np.arange(num_layers-1):
       
        # offloading AFTER this layer
        # Calculate network latency
        off_size = offload[layer_idx] 

   
        # Divide the work
        mobile_comp = 0.0
        for i in np.arange(layer_idx):
            mobile_comp += comp[i]
        server_comp = 0.0
        for j in np.arange(layer_idx, num_layers):
            server_comp += comp[j]
        
        # Calculate latency if use cpu on mobile
        cpu_latency = 0.0
        # add up the latencies
        cpu_latency = mobile_comp / mobile_cpu_tp[model] + server_comp / server_tp[model]
        # uplink lat
        cpu_latency += float(off_size) / uplink_bw
        # downlink lat, always the same
        # one floating number
        cpu_latency += 32.0 / downlink_bw 
        cpu_latency_list.append(cpu_latency)

        # and energy
        cpu_energy = mobile_comp / mobile_cpu_tp[model] * mobile_cpu_power[model] + float(off_size) / uplink_bw * uplink_power
        cpu_energy_list.append(cpu_energy)

    
        # Do the same thing for GPU
        gpu_latency = 0.0
        # add up the latencies
        gpu_latency = mobile_comp / mobile_gpu_tp[model] + server_comp / server_tp[model]
        # Calculate network latency
        off_size = offload[layer_idx] 
        # uplink lat
        gpu_latency += float(off_size) / uplink_bw
        # downlink lat, always the same
        # one floating number
        gpu_latency += 32.0 / downlink_bw 
        gpu_latency_list.append(gpu_latency)
        # and energy
        gpu_energy = mobile_comp / mobile_gpu_tp[model] * mobile_gpu_power[model] + float(off_size) / uplink_bw * uplink_power
        gpu_energy_list.append(gpu_energy)




    # Last layer. dont offload at all
    mobile_comp = 0.0
    for i in np.arange(num_layers):
        mobile_comp += comp[i]

    cpu_latency_list.append(float(mobile_comp) / mobile_cpu_tp[model])
    gpu_latency_list.append(float(mobile_comp) / mobile_gpu_tp[model])
    
    cpu_energy = float(mobile_comp) / mobile_cpu_tp[model] * mobile_cpu_power[model]
    gpu_energy = float(mobile_comp) / mobile_gpu_tp[model] * mobile_gpu_power[model]

    cpu_energy_list.append(cpu_energy)
    gpu_energy_list.append(gpu_energy)

 
    print gpu_latency_list

    print gpu_energy_list

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
    colors = ['r', 'b', 'c', 'y']

    x_values = np.arange(len(offload))

#    comp_line = ax1.bar(x_values, comp, color='r', width=width, label="Computation(GFLOP)", log=True)
    
    print layers
    print "latency"
    print cpu_latency_list
    print gpu_latency_list
    print "energy"
    print cpu_energy_list
    print gpu_energy_list
#    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

#    offload_line = ax2.bar(x_values+width, offload, color='b', width=width, label="Offloading data(MB)")

    lat_x_values = np.arange(len(offload)+1)
#    ax3.plot(lat_x_values, cpu_latency_list, color='g', linewidth=linewidth, label="mobile CPU")
#    ax3.plot(lat_x_values, gpu_latency_list, color='y', linewidth=linewidth, label="mobile GPU")
#    rects1=ax3.bar(lat_x_values, cpu_latency_list, color='g', width=width, label="mobile CPU")
    rects2=ax3.bar(lat_x_values+width, gpu_latency_list, color='y', width=width, label="mobile GPU")


    ax3.legend(title="e2e latency", loc="upper right", ncol=1, prop=fontP, bbox_to_anchor=[0.9,1.2])

#    ax1.plot(lat_x_values, cpu_energy_list, color='r', linewidth=linewidth, label="mobile CPU")
#    ax1.plot(lat_x_values, gpu_energy_list, color='b', linewidth=linewidth, label="mobile GPU")
#
#    rects3=ax1.bar(lat_x_values+2*width, cpu_energy_list, color='r', width=width, label="mobile CPU")
    rects4=ax1.bar(lat_x_values+2*width, gpu_energy_list, color='b', width=width, label="mobile GPU")

#    autolabel(ax3,rects1)
    autolabel(ax3,rects2)
#    autolabel(ax1,rects3)
    autolabel(ax1,rects4)

    ax1.legend(title="energy", loc="upper right", ncol=1, prop=fontP, bbox_to_anchor=[1.0,1.2])


    ax1.set_xlabel("layers")
    ax1.set_ylabel("Energy (J)")
#    ax2.get_yaxis().set_visible(False)
    ax3.set_ylabel("Total Latency(s)")

#    ax1.legend(bbox_to_anchor = [0.68, 1.1])
#    ax2.legend(bbox_to_anchor = [0.98, 1.1])

    layers = np.insert(layers, 0, "BEGIN")
    ax1.set_xticks(lat_x_values+2*width)
    ax1.set_xticklabels(layers, rotation=90)

    plt.savefig(model+'.png', bbox_inches='tight')
#    plt.savefig(model+'.eps', bbox_inches='tight')
#    os.popen('epstopdf '+model+'.eps')
    return 0


 
if __name__=='__main__':
    sys.exit(main(sys.argv))
