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

mobile_gpu_power = {'imc':8.7, 'face':9.5, 'vgg':8.7, 'asr':10.8, 'pos':5.2, 'ner':5.1, 'chk':5.1, 'dig':7.6} 

cpu_power_param = {}
cpu_power_param['fc'] = [0.70, 0.58, 0.68, 0.55, -0.68]
cpu_power_param['conv'] = [1.02, 0.026, 1.545, 0.018, -1.22]
cpu_power_param['act'] = 4.5

gpu_power_param = {}
gpu_power_param['fc'] = [0.71, 0.13, 0.72, 0.14, 1.21, 0.23]
gpu_power_param['conv'] = [-0.032, 0.49, 0.94, 0.31, 4.95]
gpu_power_param['act'] = 4.61
class wireless(object):
    # A class that include bw and power model of lte, 3g and wifi

    uplink_bw = {'lte':5.5, '3g':1.44, 'wifi':12.44}
    downlink_bw = {'lte':24.3, '3g':3.84, 'wifi':34.2} #FIXME wifi downlink speed slow
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

    def set_server_model(self, fname):
        self.layer_server_model = {}
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
                    parameters.append(row['p' + str(i)])
                value = (func, parameters)
                self.layer_server_model[key] = value 
        return
 
    def set_mobile_model(self, fname):
        # fname is a csv with the fitted model for each type of layer
        # layer, typeofequation, param1, param2, param3...
        self.layer_mobile_model = {}
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
        return
            
    def estimate_layer_lat(self, model, layer_in, layer_out, layer_fpops):

        func = model[0]
        params = model[1]

        if func == 'log':
            gflops = float(params[0])*math.log(float(layer_in) * float(params[1])) + float(params[2])*math.log(float(layer_out) * float(params[3])) + float(params[4])
            return float(layer_fpops) / float(gflops)
        elif func == 'linear':
            return float(layer_fpops) / (float(params[0])*layer_in + float(params[1]))
        elif func == 'dual_linear': # only for norm layer and assume local size is 5
            return float(layer_fpops) / float(params[0])*float(layer_in) + float(params[1])*5.0 + float(params[2])
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

        if layer_type == 'input' or layer_type == 'drop':
            return 0.0
        else:

            model_equation = self.layer_mobile_model[layer_type]
            return self.estimate_layer_lat(model_equation, layer_in, layer_out, layer_fpops)

    def get_server_layer_lat(self, idx):
        # get the name of the layer
        layer_name = self.layers[idx]
        layer_type = ''.join(i for i in layer_name if not (i.isdigit() or not i.isalnum()))
        layer_type = layer_type.replace(" ", "")

        # get parameters for the layer
        layer_in = self.in_dim[idx]
        layer_out = self.out_dim[idx]
        layer_fpops = self.fp_ops[idx]

        if layer_name == 'input' or layer_type == 'drop':
            return 0.0
        else:
            model_equation = self.layer_server_model[layer_type]
            return self.estimate_layer_lat(model_equation, layer_in, layer_out, layer_fpops)

    def get_layer_power(self, idx, mobile_plat):
        # get the name of the layer
        layer_name = self.layers[idx]
        layer_type = ''.join(i for i in layer_name if not (i.isdigit() or not i.isalnum()))
        layer_type = layer_type.replace(" ", "")

        # get parameters for the layer
        layer_in = self.in_dim[idx]
        layer_out = self.out_dim[idx]
        layer_fpops = self.fp_ops[idx]

        if mobile_plat == 'cpu':
            power_param = cpu_power_param
        else:
            power_param = gpu_power_param

        if layer_type == 'input' or layer_type == 'drop':
            return 0.0
        elif layer_type == 'conv' or layer_type == 'pool' or layer_type == 'local':
            param_vec = power_param['conv']
            return param_vec[0]*math.log(param_vec[1]*layer_in) + param_vec[2]*math.log(param_vec[3]*layer_out)+param_vec[4]
        elif layer_type == 'fc':
            param_vec = power_param['fc']
            return param_vec[0]*math.log(param_vec[1]*layer_in) + param_vec[2]*math.log(param_vec[3]*layer_out)+param_vec[4]
        else:
            return power_param['act']


    def get_lat_sweet_spot(self, wireless, mobile_plat, server_plat, writer):
        # explore offloading at every layer


        lat = []
        actual_lat = []
        energy = []
        mobile_lat = 0.0 #mobile lat starts at zero
        server_lat = 0.0 # server lats start at entire end-to-end
        actual_mobile_lat = 0.0
        actual_server_lat = 0.0
        

        for i in np.arange(self.num_layers()):
            server_lat += self.get_server_layer_lat(i)
            actual_server_lat += self.server_lat[i]

        for idx in np.arange(self.num_layers()):
            mobile_lat += self.get_mobile_layer_lat(idx)
            actual_mobile_lat += self.mobile_lat[idx]
            network_lat = wireless.up_lat(self.get_layer_size(idx)) 
            server_lat -= self.get_server_layer_lat(idx)
            actual_server_lat -= self.server_lat[idx]

            lat.append(mobile_lat + network_lat + server_lat)
            actual_lat.append(actual_mobile_lat + network_lat + actual_server_lat)
        
        # find lat sweet spot

        sweet_idx = np.argmin(lat)
        sweet_layer = self.layers[sweet_idx]

        # calculate real speedup
        speedup = actual_lat[0] / actual_lat[sweet_idx]

        # get actual sweet spot
        actual_sweet_idx = np.argmin(actual_lat)
        actual_sweet_layer = self.layers[actual_sweet_idx]


        opt_speedup = actual_lat[0] / actual_lat[actual_sweet_idx]

        writer.writerow([self.model, wireless.get_net_type(), mobile_plat, server_plat, sweet_layer, str(speedup), actual_sweet_layer, opt_speedup])
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

        mobile_e = 0.0
        actual_lat = 0.0
        actual_energy = []
        for idx in np.arange(self.num_layers()):
            mobile_lat += self.get_mobile_layer_lat(idx)
            actual_lat += self.mobile_lat[idx]

            network_lat = wireless.up_lat(self.get_layer_size(idx)) 

            layer_power = self.get_layer_power(idx, mobile_plat)
            
            mobile_e += layer_power * self.get_mobile_layer_lat(idx)

            actual_energy.append(actual_lat * mobile_power[self.model] + wireless.up_e(self.get_layer_size(idx)))
            energy.append(mobile_e + wireless.up_e(self.get_layer_size(idx)))
    
        print actual_energy
        print energy
        
        # find energy sweet spot
        sweet_idx = np.argmin(energy)
        sweet_layer = self.layers[sweet_idx]
        improvement = actual_energy[0] / actual_energy[sweet_idx] 

        opt_sweet_idx = np.argmin(actual_energy)
        opt_sweet_layer = self.layers[opt_sweet_idx]
        opt_improvement = actual_energy[0] / actual_energy[opt_sweet_idx]

        writer.writerow([self.model, wireless.get_net_type(), mobile_plat, sweet_layer, str(improvement), opt_sweet_layer, str(opt_improvement)])
        print 'energy - ' + 'model:' + self.model + ' sweet:'+ sweet_layer + ' reduction: '+str(improvement)

    
    def set_mobile_lat(self, fname):
        # fname is a csv that has the estimate gflops of each layer on the specific hardware
        # this should be generated by another script reading in the prototxt
        # right now this average GFLOPS per type of layer
        self.mobile_lat = []
        self.mobile_lat.append(0.0) # this is for input layer
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if self.model == row['model']:
                    self.mobile_lat.append(float(row['lat']) / 1000.0) # convert to second
    
    def set_server_lat(self, fname):
        # fname is a csv that has the estimate gflops of each layer on the specific hardware
        # this should be generated by another script reading in the prototxt
        self.server_lat= []
        self.server_lat.append(0.0) # this is for input layer
        with open(fname, 'rb') as f:
            data = csv.DictReader(f)
            for row in data:
                if self.model == row['model']:
                    layer = row['layer']
                    self.server_lat.append(float(row['lat']) / 1000.0) # conver to second


def main(args):

    model = args[1]
    layer_csv = args[2] # layer profile csv with in_size, out_size and fp_ops
    net_type = args[3] # wireless network type, lte,3g or wifi
    mobile_plat = args[4]
    mobile_model_csv = args[5] # the csv with the fitted model infos, equation types and parameters
    actual_mobile_csv = args[6]
    server_plat = args[7]
    server_model_csv =args[8]
    actual_server_csv = args[9]
    
    sweet_csv = 'lat-speedup.csv'
    writer = csv.writer(open(sweet_csv, 'a'))

    e_sweet_csv = 'e-reduction.csv'
    e_writer = csv.writer(open(e_sweet_csv, 'a'))


    net = wireless(net_type)

    nn = nn_arch(layer_csv, model)
    nn.set_mobile_lat(actual_mobile_csv)
    nn.set_server_lat(actual_server_csv)
    nn.set_mobile_model(mobile_model_csv)
    nn.set_server_model(server_model_csv)

    nn.get_lat_sweet_spot(net, mobile_plat, server_plat, writer)
    nn.get_energy_sweet_spot(net, mobile_plat, e_writer)

    return 0
    

if __name__=='__main__':
    sys.exit(main(sys.argv))
