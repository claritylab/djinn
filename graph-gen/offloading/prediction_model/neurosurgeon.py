#!/usr/bin/env python
import pdb
import copy
import sys 
import os
import csv
import math 
import numpy as np
import numpy.random as ran
import scipy
import scipy.linalg as linalg 
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as cl 

mobile_cpu_power = {'imc':6.4, 'face':6.4, 'vgg':9.2, 'asr':8.0, 'pos':5.9, 'ner':7.1, 'chk':6.3, 'dig': 3.5} 

mobile_gpu_power = {'imc':8.7, 'face':9.5, 'vgg':8.7, 'asr':12.4, 'pos':5.2, 'ner':5.1, 'chk':5.1, 'dig':7.6} 

class wireless(object):
    # A class that include bw and power model of lte, 3g and wifi

    uplink_bw = {'lte':5.5, '3g':1.44, 'wifi':12.44}
    downlink_bw = {'lte':24.3, '3g':3.84, 'wifi':21.0} #FIXME wifi downlink speed slow
    alpha_up = {'lte':438.39, '3g':868.98, 'wifi':283.17}
    alpha_down = {'lte':51.97, '3g':122.12, 'wifi':137.01}
    beta = {'lte':1288.04, '3g':817.88, 'wifi':132.86}

    def __init__(self, net_type):
        # Init the class to be one of the three networks
        self.net_type = net_type # lte, 3g or wifi
        self.up_bw = self.uplink_bw[net_type]
        self.down_bw = self.downlink_bw[net_type]
        self.up_pow = (self.alpha_up[net_type] * self.up_bw + self.beta[net_type]) / 1000.0 #watts
        self.down_pow = (self.alpha_down[net_type] * self.down_bw + self.beta[net_type]) / 1000.0 #watts

    def up_lat(self, size):
        # return uplink time
        # size should be in Mb
        return float(size) / float(self.up_bw)

    def up_e(self, size):
        # return uplink energy
        return (float(size) / float(self.up_bw)) * self.up_pow

    def get_net_type(self):
        return self.net_type

class nn_arch(object):
    # A class that parse nn.arch.in and have per layer data size and computation (# of fp ops)

    def __init__(self, fname, model):
        self.model = model
        self.layers = []
        self.data_size = []
        self.in_dim = []
        self.out_dim = []
        self.fp_ops = []
        # init the class with the nework in fname
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            done_first_layer = False
            for row in data:
                if done_first_layer == False:
                    # Need to grab input size
                    self.layers.append('input')
                    self.data_size.append(float(row['input']) * 32.0 / 1000.0 / 1000.0) # number of float -> Mb
                    #csv should have number of floating operations (FLOP)
                    self.fp_ops.append(0.0) #convert to GFLOP, no computation at input layer
                    self.in_dim.append(0.0)
                    self.out_dim.append(float(row['output']))
                    done_first_layer = True
                layer = row['layer']
                if layer in self.layers:
                    print 'duplicate entries'
                    exit(1)
                else:
                    self.layers.append(layer)
                    self.data_size.append(float(row['output']) * 32.0 / 1000.0 / 1000.0)
                    self.in_dim.append(float(row['input']))
                    self.out_dim.append(float(row['output']))
                    self.fp_ops.append(float(row['fp_ops']) / float(1000000000))
        
    def num_layers(self):
        return len(self.layers)

    def get_layer_size(self, idx):
        return self.data_size[idx]

    def get_layer_name(self, idx):
        return self.layers[idx]

    def set_model(self, fname, platform):
        # fname is a csv with the fitted model for each type of layer
        # layer, typeofequation, param1, param2, param3...
        self.layer_mobile_model = {}
        self.layer_server_model = {}
        if platform == 'mobile':
            # key is the name of the layer, values is a tuple
            # first value is the type of the function, followed by parameters
            param_count = 5
            with open(fname, 'rb') as f:
                data = csv.DictReader(f)
                for row in data:
                    key = row['layer']
                    func = row['function']
                    parameters = []
                    for i in np.arange(param_count):
                        parameters.append(row['p' + str(i)])
                    value = (func, parameters)
                    self.layer_mobile_model[key] = value
        else:
            # key is the name of the layer, values is a vector
            # first value is the type of the function, followed by parameters
            param_count = 5
            with open(fname, 'rb') as f:
                data = csv.DictReader(f)
                for row in data:
                    key = row['layer']
                    func = row['function']
                    parameters = []
                    for i in np.arange(param_count):
                        parameters.append(row['param' + str(i)])
                    value = (func, parameters)
                    self.layer_server_model[key] = value
            
#    def set_mobile_lat(self, fname):
#        # fname is a csv that has the estimate gflops of each layer on the specific hardware
#        # this should be generated by another script reading in the prototxt
#        # right now this average GFLOPS per type of layer
#        self.mobile_lat = []
#        self.mobile_lat.append(0.0) # this is for input layer
#        with open(fname, 'rb') as f:
#            data = csv.DictReader(f)
#            for row in data:
#                if self.model == row['model']:
#                    self.mobile_lat.append(float(row['lat']) / 1000.0) # convert to second
#    
#    def set_server_lat(self, fname):
#        # fname is a csv that has the estimate gflops of each layer on the specific hardware
#        # this should be generated by another script reading in the prototxt
#        self.server_lat= []
#        self.server_lat.append(0.0) # this is for input layer
#        with open(fname, 'rb') as f:
#            data = csv.DictReader(f)
#            for row in data:
#                if self.model == row['model']:
#                    layer = row['layer']
#                    self.server_lat.append(float(row['lat']) / 1000.0) # conver to second

    def estimate_layer_lat(self, model, layer_in, layer_out, layer_fpops):

        func = model[0]
        params = model[1]

        if func == 'log':
            gflops = float(params[0])*math.log(float(layer_in) * float(params[1])) + float(params[2])*math.log(float(layer_out) * float(params[3])) + float(params[4])
            return float(layer_fpops) / float(gflops) * 1000.0
        elif func == 'linear':
            return float(params[0])*layer_in + float(params[1])
        elif func == 'dual_linear': # only for norm layer and assume local size is 5
            return float(params[0])*float(layer_in) + float(params[1])*5.0 + float(params[2])
        else:
            return 0.0
   
    def get_mobile_layer_lat(self, idx):
        # get the name of the layer
        layer_name = self.layers[idx]
        layer_type = ''.join(i for i in layer_name if not (i.isdigit() or not i.isalnum()))
        layer_type = layer_type.replace(" ", "")

        # get parameters for the layer
        layer_in = self.in_dim[idx]
        layer_out = self.out_dim[idx]
        layer_fpops = self.fp_ops[idx]

        if layer_type == 'input' or layer_type == 'drop' or layer_type == 'prob'  or layer_type=='argmax':
            return 0.0
        elif layer_type == 'conv' or layer_type == 'fc' or layer_type == 'sigmoid' or layer_type == 'relu' or layer_type == 'pool' or layer_type == 'norm':

            model_equation = self.layer_mobile_model[layer_type]
            return self.estimate_layer_lat(model_equation, layer_in, layer_out, layer_fpops)

    def get_server_layer_lat(self, idx):
        # get the name of the layer
        layer_name = self.layers[idx]
        layer_type = ''.join(i for i in layer_name if not i.isdigit())

        # get parameters for the layer
        layer_in = self.in_dim[idx]
        layer_out = self.out_dim[idx]
        layer_fpops = self.fp_ops[idx]

        if layer_name == 'input':
            return 0.0
        else:
            model_equation = self.layer_server_model[layer_type]
            return estimate_layer_lat(model_equation, layer_in, layer_out, layer_fpops)

    def get_lat_sweet_spot(self, wireless, mobile_plat, server_plat, writer):
        # explore offloading at every layer
        lat = []
        energy = []
        mobile_lat = 0.0 #mobile lat starts at zero
        server_lat = 0.0 # server lats start at entire end-to-end
        for i in np.arange(self.num_layers()):
            server_lat += self.get_server_layer_lat(i)

        for idx in np.arange(self.num_layers()):
            mobile_lat += self.get_mobile_layer_lat(idx)
            network_lat = wireless.up_lat(self.get_layer_size(idx)) 
            server_lat -= self.get_server_layer_lat(idx)

            lat.append(mobile_lat + network_lat + server_lat)
        
        # find lat sweet spot
        sweet_idx = np.argmin(lat)
        sweet_layer = self.layers[sweet_idx]
        sweet_lat = lat[sweet_idx]
        speedup = lat[0] / sweet_lat

        writer.writerow([self.model, wireless.get_net_type(), mobile_plat, server_plat, sweet_layer, str(speedup)])
        print 'latency - ' + 'model:' + self.model + ' sweet:'+ sweet_layer + ' speedup: '+str(speedup)

    def get_energy_sweet_spot(self, wireless, mobile_plat, writer):
        # explore offloading at every layer
        energy = []
        mobile_lat = 0.0 #mobile lat starts at zero
        
        if mobile_plat == 'cpu':
            mobile_power = mobile_cpu_power
        elif mobile_plat == 'gpu':
            mobile_power = mobile_gpu_power
        else:
            print 'unsupported mobile hardware'
            exit(1)

        for idx in np.arange(self.num_layers()):
            mobile_lat += self.get_mobile_layer_lat(idx)
            network_lat = wireless.up_lat(self.get_layer_size(idx)) 

            energy.append(mobile_lat * mobile_power[self.model] + wireless.up_e(self.get_layer_size(idx)))
        
        # find energy sweet spot
        sweet_idx = np.argmin(energy)
        sweet_layer = self.layers[sweet_idx]
        sweet_e = energy[sweet_idx]
        improvement = energy[0] / sweet_e

        writer.writerow([self.model, wireless.get_net_type(), mobile_plat, sweet_layer, str(improvement)])
        print 'energy - ' + 'model:' + self.model + ' sweet:'+ sweet_layer + ' reduction: '+str(improvement)

    
def main(args):

    model = args[1]
    layer_csv = args[2] # layer profile csv with in_size, out_size and fp_ops
    net_type = args[3] # wireless network type, lte,3g or wifi
    mobile_plat = args[4]
    mobile_model_csv = args[5] # the csv with the fitted model infos, equation types and parameters
#    mobile_lat_csv = args[5] # per layer GFLOPS predicition csv, generated by another script for a specific hardware
#    server_plat = args[6]
#    server_model_csv =args[7]
#    server_lat_csv = args[7]

    sweet_csv = 'lat-speedup.csv'
    writer = csv.writer(open(sweet_csv, 'a'))

    e_sweet_csv = 'e-reduction.csv'
    e_writer = csv.writer(open(e_sweet_csv, 'a'))


    net = wireless(net_type)

    nn = nn_arch(layer_csv, model)
#    nn.set_mobile_lat(mobile_lat_csv)
#    nn.set_server_lat(server_lat_csv)
    nn.set_model(mobile_model_csv, 'mobile')
#    nn.set_model(server_model_csv, 'server')

    for i in np.arange(nn.num_layers()):
        print str(nn.get_layer_name(i)) + " " + str(nn.get_mobile_layer_lat(i))
    nn.get_lat_sweet_spot(net, mobile_plat, server_plat, writer)
#    nn.get_energy_sweet_spot(net, mobile_plat, e_writer)

    return 0
    

if __name__=='__main__':
    sys.exit(main(sys.argv))
