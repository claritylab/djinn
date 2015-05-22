#! /usr/bin/env python
import sys
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats


#fitting functions
def func_log (x, a, b, c, d, e):
     return (a*np.log(x[0]+b)) + (c*np.log(x[1]+d)) + e

def func_exp (x, a, b, c, d, e):
     return b*np.exp(a*(x[0])) + d*np.exp(c*(x[1])) + e

def func_quad (x, a, b, c, d, e):
     return a*(x[0]**2) + b*(x[1]**2) + c*x[0] + d*x[1] + e

def func_lin (x, a, b):
     return a*x + b

def main(args):
    #PARAMETER
    #----------------------
    #input file name
    input_file = args[1] 
    
    #number of steps in fitted line
    steps=1000.0
    #----------------------
    
    #initialize vectors
    in_dim = []
    out_dim = []
    gflops = []
    
    #open and read input file
    with open(input_file, 'rb') as f:
         data = csv.DictReader(f)
         for row in data:
               gflops.append(float(row['gflops']))
               in_dim.append(float(row['channel']) * float(row['width']) * float(row['height']))
    
    x_data = in_dim
    y_data = gflops
    
    #perform curve fit
    popt, pcov = curve_fit(func_lin,x_data, y_data)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data,y_data)
    
    #find residuals (distance from each fitted point to actual point)
    residuals = []
    for x in range (0, len(x_data)):
         residuals.append(y_data[x] - func_lin(x_data[x], popt[0], popt[1]))
    
         #restrict to only predicting GFLOPS >= 0
         if residuals[-1] < 0:
              residuals[-1] = 0
    
    
    #calculate standard error (average distance from fitted point to actual point)
    s_err = np.mean([abs(x) for x in residuals])
    
    csv_line = 'softmax,linear,a*x+b,2,' + str(popt[0]) + ',' + str(popt[1]) + ',' + str(s_err)+','+str(r_value)
    print csv_line

if __name__=='__main__':
    sys.exit(main(sys.argv))

##initialize variables
#fit_x = []
#fit_x0 = []
#fit_x1 = []
#fit_y = []
#
## Distribute x-vars over x-range (GFLOPS is x-axis)
##for val in np.arange(0, fp_ops[-1], fp_ops[-1]/(steps+1)):
##    fit_x.append(val)
#
#for val in np.arange(0, out_dim[-1], out_dim[-1]/(steps+1)):
#    fit_x1.append(val)
#
#fit_x0 = int(steps+1) * [128]
#
#fit_x0 = int(steps+1) * [128]
#fit_y = []
#for x in range (0, len(fit_x0)):
#     fit_y.append(func_lin(fit_x1[x], popt[0], popt[1]))
#     if fit_y[-1] < 0:
#          fit_y[-1] = 0
#plt.plot(fit_x1,fit_y)
#
#fit_x0 = int(steps+1) * [512]
#fit_y = []
#for x in range (0, len(fit_x0)):
#     fit_y.append(func_lin(fit_x1[x], popt[0], popt[1]))
#plt.plot(fit_x1,fit_y)
#
#fit_x0 = int(steps+1) * [2048]
#fit_y = []
#for x in range (0, len(fit_x0)):
#     fit_y.append(func_lin(fit_x1[x], popt[0], popt[1]))
#plt.plot(fit_x1,fit_y)
#
#fit_x0 = int(steps+1) * [8192]
#fit_y = []
#for x in range (0, len(fit_x0)):
#     fit_y.append(func_lin(fit_x1[x], popt[0], popt[1]))
#plt.plot(fit_x1,fit_y)
#
##plot data
#plt.plot(x_data, y_data, 'ro')
#plt.title('Rectified Linear Layer')
#plt.ylabel('GFLOPS')
#plt.xlabel('Number of outputs')
##plt.plot(fit_x1,fit_y)
#plt.show()
