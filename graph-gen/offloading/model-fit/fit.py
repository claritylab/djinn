#! /usr/bin/env python

import numpy as np
import math
import csv
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


#fitting functions
def func_log (x, a, b, c, d, e):
     return (a*np.log(x[0]+b)) + (c*np.log(x[1]+d)) + e

def func_exp (x, a, b, c, d, e):
     return b*np.exp(a*(x[0])) + d*np.exp(c*(x[1])) + e

def func_quad (x, a, b, c, d, e):
     return a*(x[0]**2) + b*(x[1]**2) + c*x[0] + d*x[1] + e

def func_lin (x, a, b, c, d, e):
     return a*x + b

#PARAMETERS
#----------------------
#input file name
input_file = 'fc.csv'

#number of steps in fitted line
steps=1000.0
#----------------------


#initialize vectors
gflops = []
fp_ops = []
in_dim = []
out_dim = []

#open and read input file
with open(input_file, 'rb') as f:
     data = csv.DictReader(f)
     for row in data:
           gflops.append(float(row['GFLOPS']))
           fp_ops.append(float(row['fp_ops']))
           in_dim.append(float(row['in_dim']))
           out_dim.append(float(row['out_dim']))

x_data = [in_dim, out_dim]
y_data = gflops

#perform curve fit
popt, pcov = curve_fit(func_log,x_data, y_data)
print 'optimal parameters:'
print popt
print 'covarience matrix:'
print pcov

#find residuals (distance from each fitted point to actual point)
residuals = []
for x in range (0, len(x_data[0])):
     residuals.append(y_data[x] - func_log([x_data[0][x], x_data[1][x]], popt[0], popt[1], popt[2], popt[3], popt[4]))

     #restrict to only predicting GFLOPS >= 0
     if residuals[-1] < 0:
          residuals[-1] = 0

#calculate standard error (average distance from fitted point to actual point)
print 'standard error:'
print type(residuals[0])
s_err = np.mean([abs(x) for x in residuals])
print s_err

#initialize variables
fit_x = []
fit_x0 = []
fit_x1 = []
fit_y = []

# Distribute x-vars over x-range (GFLOPS is x-axis)
#for val in np.arange(0, fp_ops[-1], fp_ops[-1]/(steps+1)):
#    fit_x.append(val)

for val in np.arange(0, out_dim[-1], out_dim[-1]/(steps+1)):
    fit_x1.append(val)

fit_x0 = int(steps+1) * [128]

#fit_y = func_log([fit_x1, fit_x0], popt[0], popt[1], popt[2], popt[3], popt[4])

fit_x0 = int(steps+1) * [128]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
     if fit_y[-1] < 0:
          fit_y[-1] = 0
plt.plot(fit_x1,fit_y)

fit_x0 = int(steps+1) * [512]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
plt.plot(fit_x1,fit_y)

fit_x0 = int(steps+1) * [2048]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
plt.plot(fit_x1,fit_y)

fit_x0 = int(steps+1) * [8192]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_log([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
plt.plot(fit_x1,fit_y)

#plot data
plt.plot(x_data[1], y_data, 'ro')
plt.title('Fully Connected Layer')
plt.ylabel('GFLOPS')
plt.xlabel('Number of outputs')
#plt.plot(fit_x1,fit_y)
plt.show()

# relu ----------------------------------------------------------------------------------
#initialize vectors
in_dim = []
out_dim = []
gflops = []

#open and read input file
with open(input_file, 'rb') as f:
     data = csv.DictReader(f)
     for row in data:
           gflops.append(float(row['GFLOPS']))
           in_dim.append(float(row['in_dim']))
           out_dim.append(float(row['out_dim']))

x_data = out_dim
y_data = gflops

#perform curve fit
popt, pcov = curve_fit(func_lin,x_data, y_data)
print 'optimal parameters:'
print popt
print 'covarience matrix:'
print pcov

#temp fix because im lazy
popt = np.append(popt, [0,0,0])

#find residuals (distance from each fitted point to actual point)
residuals = []
for x in range (0, len(x_data)):
     residuals.append(y_data[x] - func_lin(x_data, popt[0], popt[1], popt[2], popt[3], popt[4]))

     #restrict to only predicting GFLOPS >= 0
     if residuals[-1] < 0:
          residuals[-1] = 0

#calculate standard error (average distance from fitted point to actual point)
print 'standard error:'
print type(residuals[0])
s_err = np.mean([abs(x) for x in residuals])
print s_err

#initialize variables
fit_x = []
fit_x0 = []
fit_x1 = []
fit_y = []

# Distribute x-vars over x-range (GFLOPS is x-axis)
#for val in np.arange(0, fp_ops[-1], fp_ops[-1]/(steps+1)):
#    fit_x.append(val)

for val in np.arange(0, out_dim[-1], out_dim[-1]/(steps+1)):
    fit_x1.append(val)

fit_x0 = int(steps+1) * [128]

fit_x0 = int(steps+1) * [128]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_lin(fit_x1[x], popt[0], popt[1], popt[2], popt[3], popt[4]))
     if fit_y[-1] < 0:
          fit_y[-1] = 0

print len(fit_x1)
print len(fit_y)

plt.plot(fit_x1,fit_y)

fit_x0 = int(steps+1) * [512]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_lin([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
plt.plot(fit_x1,fit_y)

fit_x0 = int(steps+1) * [2048]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_lin([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
plt.plot(fit_x1,fit_y)

fit_x0 = int(steps+1) * [8192]
fit_y = []
for x in range (0, len(fit_x0)):
     fit_y.append(func_lin([fit_x1[x], fit_x0[x]], popt[0], popt[1], popt[2], popt[3], popt[4]))
plt.plot(fit_x1,fit_y)

#plot data
plt.plot(x_data[1], y_data, 'ro')
plt.title('Fully Connected Layer')
plt.ylabel('GFLOPS')
plt.xlabel('Number of outputs')
#plt.plot(fit_x1,fit_y)
plt.show()