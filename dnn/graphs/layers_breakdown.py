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
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as cl

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 

fsize='xx-small'
fontP = FontProperties()
fontP.set_size(fsize)
width = 0.2
linewidth = 2
#colors = ['r', 'b', 'c', 'y']

#server_tp = {'imc':352.45, 'face':49.50, 'vgg':1008.22, 'asr':3.55, 'pos':1.48, 'ner':1.57, 'chk':1.47} 
#mobile_gpu_tp = {'imc':23.15, 'face':5.34, 'vgg':49.42, 'asr':0.18, 'pos':0.32, 'ner':0.34, 'chk':0.29} 
#mobile_cpu_tp = {'imc':5.88, 'face':1.36, 'vgg':4.00, 'asr':0.017, 'pos':0.22, 'ner':0.29, 'chk':0.22} 
mobile_cpu_power = {'imc':6.4, 'face':6.4, 'vgg':9.2, 'asr':8.0, 'pos':5.9, 'ner':7.1, 'chk':6.3} 
mobile_gpu_power = {'imc':8.7, 'face':9.5, 'vgg':8.7, 'asr':12.4, 'pos':5.2, 'ner':5.1, 'chk':5.1} 

#########LTE#######
lte_uplink_bw = 5.5 #LTE Mbps
lte_downlink_bw = 24.3 #LTE Mbps
lte_uplink_power = (438.39 * lte_uplink_bw + 1288.04) / 1000.0 # LTE W
lte_downlink_power = (51.97 * lte_downlink_bw + 1288.04) / 1000.0 # LTE W

#######3g###########
threeg_uplink_bw = 1.44
threeg_downlink_bw = 3.84
threeg_uplink_power = (868.98 * threeg_uplink_bw + 817.88) / 1000.0 # 3g W
threeg_downlink_power = (122.12 * threeg_downlink_bw + 817.88) / 1000.0 # 3g W

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


def breakdown(model, platform, network, model_csv, mobile_csv, server_csv, mobile_power, plt, output_csv):

    # Pick the right network model
    if network == "LTE":
        uplink_bw = lte_uplink_bw
        downlink_bw = lte_downlink_bw
        uplink_power = lte_uplink_power
        downlink_power = lte_downlink_power
    elif network == "3G":
        uplink_bw = threeg_uplink_bw
        downlink_bw = threeg_downlink_bw
        uplink_power = threeg_uplink_power
        downlink_power = threeg_downlink_power
    else:
        print "network type not supported"
        exit(1)
    
    # Read in measured mobile latency
    reader=csv.reader(open(mobile_csv, 'rb'), delimiter=',')
    mobile_lat_mat = np.array(list(reader))
    mobile_lat_mat = np.delete(mobile_lat_mat,0,0)
    mobile_lat = mobile_lat_mat[:,1]
    mobile_lat = mobile_lat.astype(float) / 1000.0 # Convert to seconds

    # Read in offload data size
    reader = csv.reader(open(model_csv, 'rb'), delimiter=',')
    model_mat = np.array(list(reader))
    model_mat = np.delete(model_mat, 0, 0)
    # Get the layers
    layers = model_mat[:,0]
    # Get the data to be offload
    offload = model_mat[:,2]
    offload = offload.astype(float)
    offload = offload * 32.0 / 1024.0 / 1024.0 #covnert to Mb

    # Get the depth of network
    num_layers = len(offload)
    mobile_lat_list = []
    server_lat_list = []
    offload_lat_list = []
    
    mobile_energy_list = []
    server_energy_list = []
    offload_energy_list = []
 
    # Read in measured server latency 
    reader=csv.reader(open(server_csv, 'rb'), delimiter=',')
    server_lat_mat = np.array(list(reader))
    server_lat_mat = np.delete(server_lat_mat,0,0)
    server_lat = server_lat_mat[:,1]
    server_lat = server_lat.astype(float) / 1000.0

    # CSV reading all done, start analysis
    # if offload at the very beginning
    data_before_first_layer = float(model_mat[0][1]) * 32.0 /1024.0 /1024.0  

    cur_mobile_lat = 0.0
    cur_offload_lat = float(data_before_first_layer)/uplink_bw 
    cur_server_lat = 0.0
    for i in np.arange(num_layers):
        cur_server_lat += server_lat[i]
    
    cur_mobile_energy = 0.0
    cur_offload_energy = cur_offload_lat * uplink_power 
    cur_server_energy = 0.0 #FIXME
    
    mobile_lat_list.append(cur_mobile_lat)
    server_lat_list.append(cur_server_lat)
    offload_lat_list.append(cur_offload_lat)
    
    mobile_energy_list.append(cur_mobile_energy)
    server_energy_list.append(cur_server_energy)
    offload_energy_list.append(cur_offload_energy)
    
    # Sweep through the network to find the sweep spot
    for layer_idx in np.arange(num_layers-1):
       
        # offloading AFTER this layer
        # Calculate network latency
        cur_offload_size = offload[layer_idx] 
   
        # Divide the work
        cur_mobile_lat = 0.0
        for i in np.arange(layer_idx+1):
            cur_mobile_lat += mobile_lat[i]

        cur_server_lat = 0.0
        for j in np.arange(layer_idx+1, num_layers):
            cur_server_lat += server_lat[j]
        
        cur_offload_lat = float(cur_offload_size) / uplink_bw

        cur_mobile_energy = cur_mobile_lat * mobile_power[model]
        cur_offload_energy = cur_offload_size * uplink_power
        cur_server_energy = 0.0 #FIXME
    
        mobile_lat_list.append(cur_mobile_lat)
        server_lat_list.append(cur_server_lat)
        offload_lat_list.append(cur_offload_lat)
       
        mobile_energy_list.append(cur_mobile_energy)
        server_energy_list.append(cur_server_energy)
        offload_energy_list.append(cur_offload_energy)
   
    # Last layer. dont offload at all
    cur_mobile_lat = 0.0
    for i in np.arange(num_layers):
        cur_mobile_lat += mobile_lat[i]
    cur_offload_lat = 0
    cur_server_lat = 0

    cur_mobile_energy = cur_mobile_lat * mobile_power[model]
    cur_server_energy = 0
    cur_offload_energy = 0


    mobile_lat_list.append(cur_mobile_lat)
    server_lat_list.append(cur_server_lat)
    offload_lat_list.append(cur_offload_lat)
       
    mobile_energy_list.append(cur_mobile_energy)
    server_energy_list.append(cur_server_energy)
    offload_energy_list.append(cur_offload_energy)

    # PLOT
    fig,ax1 = plt.subplots()
    fig.set_size_inches(15,5)

    x_values = np.arange(len(offload) + 1)
    # plot latency first
    colors=color_maker(3, min=0.1, max=0.5)
    
    ax1.bar(x_values, mobile_lat_list, width, color=colors[0], label="mobile")
    ax1.bar(x_values, offload_lat_list, width, color=colors[1], label="network", bottom=mobile_lat_list)
    ax1.bar(x_values, server_lat_list, width, color=colors[2], label="server", bottom=np.add(mobile_lat_list, offload_lat_list))
    ax1.legend(title="latency", ncol=1, prop=fontP, bbox_to_anchor=[0.8,1.2])

    # plot energy
    colors=color_maker(2, min=0.55, max=0.9)
    ax2 = ax1.twinx()
    ax2.bar(x_values+width, mobile_energy_list, width, color=colors[0], label="mobile")
    ax2.bar(x_values+width, offload_energy_list, width, color=colors[1], label="network", bottom=mobile_energy_list)
#    ax2.bar(x_values+width, server_energy_list, width, color=colors[2], label="server", bottom=np.add(mobile_energy_list, offload_energy_list))
    ax2.legend(title="energy", ncol=1, prop=fontP, bbox_to_anchor=[1.0,1.2])

    # lable x-axis
    ax1.set_xlabel("layers")
    ax1.set_ylabel("End-to-end Latency (s)")
    ax2.set_ylabel("Energy (J)")

    layers = np.insert(layers, 0, "BEGIN")
    ax1.set_xticks(x_values+width)
    ax1.set_xticklabels(layers, rotation=90)
    ax1.set_xlim([-1, len(offload)+1])

    # Find the sweet spot to offload and write to csv
    writer=csv.writer(open(output_csv, 'wb'), delimiter=',')
    # Find the sweet spot in terms of energy and latency
    e2e_lat_list = np.add(np.add(mobile_lat_list,offload_lat_list), server_lat_list)
    energy_list = np.add(mobile_energy_list, offload_energy_list)


    lat_sweet_spot = layers[np.argmin(e2e_lat_list)]
    energy_sweet_spot = layers[np.argmin(energy_list)]

    # write to csv
    # model, platform, network, target(lat or energy), offloading_point
    writer.writerow([model, platform, network, "e2e_latency", lat_sweet_spot])
    writer.writerow([model, platform, network, "mobile_energy", energy_sweet_spot])

    return


def main(argv):
 
    model = argv[1]
    platform = argv[2] # cpu, gpu
    network = argv[3] # LTE, 3G

    print "model: "+model+" platform: "+platform+" network: "+network
    
    model_csv = "layers_"+model+".csv"
    server_csv = "lats/k40_"+model+"_layer.csv"
    mobile_csv = "lats/"+platform+"_"+model+"_layer.csv"

    # open up a output csv
    output_csv = "stacked/"+model+"_sweet.csv"
#    header_writer = csv.writer(open(output_csv, 'wb'), delimiter=',')
#    header_writer.writerow(['model', 'platform', 'network', 'target', 'sweet_spot'])

    # Plot
    import matplotlib
    from matplotlib import pyplot as plt
    
    plt.rc("font", family="serif")
    plt.rc("font", size=15)
    plt.rc('legend',fontsize=15)

    breakdown(model, platform, network, model_csv, mobile_csv, server_csv, mobile_gpu_power, plt, output_csv)

    plt.savefig("stacked/"+model+"_"+platform+"_"+network+'.png', bbox_inches='tight')
    return 0

if __name__=='__main__':
    sys.exit(main(sys.argv))
