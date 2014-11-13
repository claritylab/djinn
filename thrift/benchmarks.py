#!/usr/bin/env python

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cmx
import matplotlib.colors as cl

from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter

plt.rc("font", family="serif")
plt.rc("font", size=12)
plt.rc('legend', fontsize=11)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9, 7)

"""
Function to format percentage numbers
"""
def to_percentage(y, position):
  v = str(100 * y)
  if matplotlib.rcParams["text.usetex"] is True:
    return v + r"$\%$"
  else:
    return v + "%"

# map is the name of one of the colormaps from
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
  assert(min >= 0.000 and max <= 1.000 and max > min)
  gran = 100000.0
  maker = cmx.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=int(gran)),
                             cmap=plt.get_cmap(map))
  r = [min * gran]
  if count > 1:
    r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
  return [maker.to_rgba(t) for t in r]

def parse_tcpdump(filename):
  fp = open(filename)
  x_values = []
  y_values = []
  for line in fp:
    items = line.strip().split()
    x_values.append(int(items[1]))
    y_values.append(float(items[0]))
  return x_values, y_values

# treadmill
treadmill_latency_cdf = [0] * 1000000
fp = open("log_treadmill_1_10_60.json")
treadmill_json = json.loads(fp.read())
latency_histogram = treadmill_json["request_latency"]["histogram"]
fp.close()
x_values, y_values = zip(*sorted(latency_histogram.items(), key=lambda x: float(x[0])))
total = float(sum(y_values))
pdf = map(lambda x: x / total, y_values)
# Comment the following line out if you want PDF instead of CDF
cdf = np.cumsum(pdf)
x_values = [float(r) for r in x_values]
x_values.insert(0, 0)
x_values.append(1000000)
cdf = np.insert(cdf, 0, 0.0)
cdf = np.append(cdf, 1.0)
treadmill_latency_cdf = interp1d(x_values, cdf)

"""
treadmill_tcpdump_x_values, treadmill_tcpdump_y_values \
    = parse_tcpdump("treadmill_tcpdump.dist")
"""

# plot the cdf
color_list = color_maker(2, map="gnuplot")

ax = plt.subplot(3, 1, 1)
x_axis = range(0, 1000000)
"""
ax.plot(loader_latency_cdf(x_axis), color=color_list[0],
        lw=1, label="CloudSuite")
p99 = next(i for i, v in enumerate(loader_latency_cdf(x_axis)) if v >= 0.99)
ax.vlines(p99, 0, 1, colors=color_list[0], linestyle="dashed",
          label="P99 CloudSuite")
ax.plot(loader_tcpdump_x_values, loader_tcpdump_y_values, color=color_list[1],
        lw=1, label="tcpdump")
p99 = next(i for i, v in enumerate(loader_tcpdump_y_values) if v >= 0.99)
ax.vlines(p99, 0, 1, colors=color_list[1], linestyle="dashed",
          label="P99 tcpdump")
ax.set_xlabel("Latency (us)")
ax.set_ylabel("CDF")
ax.grid()
ax.set_ylim(0, 1)
ax.set_xlim(0, 250)
ax.legend(loc="lower right")

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.80])

ax = plt.subplot(3, 1, 2)
ax.plot(x_axis, mutilate_latency_cdf, color=color_list[0],
        lw=1, label="Mutilate")
p99 = next(i for i, v in enumerate(mutilate_latency_cdf) if v >= 0.99)
ax.vlines(p99, 0, 1, colors=color_list[0], linestyle="dashed",
          label="P99 Mutilate")
ax.plot(mutilate_tcpdump_x_values, mutilate_tcpdump_y_values,
        color=color_list[1], lw=1, label="tcpdump")
p99 = next(i for i, v in enumerate(mutilate_tcpdump_y_values) if v >= 0.99)
ax.vlines(p99, 0, 1, colors=color_list[1], linestyle="dashed",
          label="P99 tcpdump")
ax.set_xlabel("Latency (us)")
ax.set_ylabel("CDF")
ax.grid()
ax.set_ylim(0, 1)
ax.set_xlim(0, 250)
ax.legend(loc="lower right")

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.80])
"""

ax = plt.subplot(1, 1, 1)
ax.plot(treadmill_latency_cdf(x_axis), color=color_list[0],
        lw=1, label="Treadmill")
p99 = next(i for i, v in enumerate(treadmill_latency_cdf(x_axis)) if v >= 0.99)
ax.vlines(p99, 0, 1, colors=color_list[0], linestyle="dashed",
          label="P99 Treadmill")
"""
ax.plot(treadmill_tcpdump_x_values, treadmill_tcpdump_y_values,
        color=color_list[1], lw=1, label="tcpdump")
p99 = next(i for i, v in enumerate(treadmill_tcpdump_y_values) if v >= 0.99)
ax.vlines(p99, 0, 1, colors=color_list[1], linestyle="dashed",
          label="P99 tcpdump")
"""
ax.set_xlabel("Latency (us)")
ax.set_ylabel("CDF")
ax.grid()
ax.set_ylim(0, 1)
ax.set_xlim(0, 250)
ax.legend(loc="lower right")

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.20,
                 box.width, box.height * 0.80])

plt.savefig("benchmarkslow.pdf")
