#! /usr/bin/env python
import sys
import numpy as np
import scipy
import math
import csv
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats


#fitting functions
def func_log (x, a, b, d, e, g):
     return (a*np.log(b*x[0])) + (d*np.log(e*x[1])) + g

#def func_log (x, a, b, g):
#     return a*np.log(b*x[0]) + g

def func_exp (x, a, b, c, d, e):
     return b*np.exp(a*(x[0])) + d*np.exp(c*(x[1])) + e

def func_quad (x, a, b, c, d, e):
     return a*(x[0]**2) + b*(x[1]**2) + c*x[0] + d*x[1] + e

def func_lin (x, a, b, c, d, e):
     return a*x + b

def main(args):
    #PARAMETERS
    #----------------------
    #input file name
    input_file = args[1] 
    
    #number of steps in fitted line
    steps=1000.0
    #----------------------
    
    #initialize vectors
    gflops = []
    fp_ops = []
    in_dim = []
    out_dim = []
    channel = []
    height = []
    width = []
    num_output = []
    kernel_size = []
    lat = []
    stride = []
    
    #open and read input file
    with open(input_file, 'rb') as f:
         data = csv.DictReader(f)
         for row in data:
               gflops.append(float(row['gflops']))
               fp_ops.append(float(row['fpops']))
               lat.append(float(row['lat']))
               #in_dim.append(float(row['in_dim']))
               num_output.append(float(row['num_output']))
               out_dim.append(float(row['out_dim']))
               channel.append(float(row['channel']))
               height.append(float(row['height']))
               width.append(float(row['width']))
               kernel_size.append(float(row['kernel_size']))
               stride.append(float(row['stride']))
    
    in_dim = np.multiply(np.multiply(channel, height), width)
    
    out_dim =  np.multiply(np.multiply(out_dim, out_dim), num_output)
    
    
    #x_data = [in_dim, num_output, kernel_size, stride]
    x_data = [in_dim, out_dim]
    y_data = gflops 
    
    #perform curve fit
    popt, pcov = curve_fit(func_log, x_data, y_data)

    s_res = np.dot((y_data - func_log(x_data, *popt)),(y_data - func_log(x_data, *popt)))
    ymean = np.mean(y_data)
    ss_tot = np.dot((y_data-ymean),(y_data-ymean))
    
    #find residuals (distance from each fitted point to actual point)
    residuals = []
    for x in range (0, len(x_data[0])):
         residuals.append(y_data[x] - func_log([x_data[0][x], x_data[1][x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
    
         #restrict to only predicting GFLOPS >= 0
         if residuals[-1] < 0:
              residuals[-1] = 0
    
    #calculate standard error (average distance from fitted point to actual point)
    s_err = np.mean([abs(x) for x in residuals])
    csv_line = 'conv,log,a*log(b*x0)+c*log(d*x1)+e,5,'
    for i in np.arange(5):
        csv_line += str(popt[i])+','
    csv_line += str(s_err) + ','
    r_sq = 'NA'
    csv_line += str(r_sq)
    
    print csv_line
    
##########################
    return
##########################
    
    #initialize variables
    fit_x = []
    fit_x0 = []
    fit_x1 = []
    fit_y = []
    
    # Distribute x-vars over x-range (GFLOPS is x-axis)
    for val in np.arange(0, fp_ops[-1], fp_ops[-1]/(steps+1)):
        fit_x.append(val)
    
    for val in np.arange(0, in_dim[-1], in_dim[-1]/(steps+1)):
        fit_x1.append(val)
    
    fit_x0 = int(steps+1) * [2]
    fit_y = []
    for x in range (0, len(fit_x0)):
         fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
         if fit_y[-1] < 0:
              fit_y[-1] = 0
    plt.plot(fit_x1,fit_y)
    
    fit_x0 = int(steps+1) * [3]
    fit_y = []
    for x in range (0, len(fit_x0)):
         fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
    plt.plot(fit_x1,fit_y)
    
    fit_x0 = int(steps+1) * [4]
    fit_y = []
    for x in range (0, len(fit_x0)):
         fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
    plt.plot(fit_x1,fit_y)
    
    fit_x0 = int(steps+1) * [90000]
    fit_y = []
    for x in range (0, len(fit_x0)):
         fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
    plt.plot(fit_x1,fit_y)
    
    #plot data
    plt.plot(x_data[0], y_data, 'ro')
    plt.title('Convolutional Layer')
    plt.ylabel('GFLOPS')
    plt.xlabel('Number of inputs')
    plt.plot(fit_x1,fit_y)
    # plt.show()

if __name__=='__main__':
    sys.exit(main(sys.argv))
