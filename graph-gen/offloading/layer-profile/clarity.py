import sys, os, csv, re, random
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as pl
import matplotlib.markers as mks

# map is the name of one of the colormaps from
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
def color_maker(count, map='gnuplot2', min=0.100, max=0.900):
    assert(min >= 0.000 and max <= 1.000 and max > min)
    gran = 100000.0
    maker = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=int(gran)),
                               cmap=pl.get_cmap(map))
    r = [min * gran]
    if count > 1:
        r = [min * gran + gran * x * (max - min) / float(count - 1) for x in range(0, count)]
        return [maker.to_rgba(t) for t in r]

def marker_maker(count, randomize=False):
    """ makes a list of markers of length count """
    markers = mks.MarkerStyle.markers
    del markers[0]
    num = len(markers.values()) # hack?
    m = [str(m) for m in markers]
    if randomize is True:
        return [ m[random.randint(0, num - 1)] for i in range(0, count)]
    else:
        return [ m[i] for i in range(0, count)]

def color_variant(hex_color, brightness_offset=10):  
    """ takes a color like #87c95f and produces a lighter or darker variant """  
    if len(hex_color) != 7:
        return hex_color
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    # make sure new values are between 0 and 255
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])

def get_num(x):
    return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))

def get_data(filename, field=""):
    with open(filename, 'rb') as f:
        d = csv.DictReader(f)
        for row in d:
            data.append( row );

    return data
