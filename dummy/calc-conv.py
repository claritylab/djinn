#!/usr/bin/env python

import math
import subprocess, re, os, sys, csv

featmaps = {}     #  c   h/w  out   k  s
featmaps['input']   = [3,  227, 96,  11, 4]
featmaps['input11'] = [3,  227, 256, 11, 2]
featmaps['input12'] = [3,  227, 48,  11, 1]
# FACE
featmaps['input71'] = [3,  152, 32,  11, 1]
# featmaps['input72'] = [32,  71, 32,  3,  2]
featmaps['input72'] = [64,  71, 32,  5,  2]
featmaps['input73'] = [32,  64, 16,  9,  2]
featmaps['input61'] = [3,  152, 256, 9, 2]
featmaps['input62'] = [3,  152, 48,  9, 1]

# DIG
featmaps['input60'] = [1, 28, 20,  5, 1]
featmaps['input60'] = [1, 22, 50,  5, 1]

featmaps['input61'] = [1, 28, 20,  7, 1]

featmaps['input20'] = [96,  55, 256, 5, 1]
featmaps['input21'] = [96,  55, 128, 5, 2]
featmaps['input22'] = [96,  55,  64, 5, 4]
#
# featmaps['input30'] = [256, 27, 384, 3, 1]
# # featmaps['input31'] = [256, 27, 256, 3, 2]
# # featmaps['input32'] = [256, 27, 512, 3, 4]
#
featmaps['input40'] = [384, 13, 256, 3, 1]
# featmaps['input41'] = [384, 13, 256, 3, 2]
# # featmaps['input42'] = [384, 13, 256, 3, 4]
#
featmaps['input50'] = [256, 13, 256, 3, 1]
# featmaps['input51'] = [256, 13, 256, 3, 1]
# featmaps['input52'] = [256, 13, 256, 3, 1]
# featmaps['input53'] = [256, 13, 256, 3, 1]
# featmaps['input54'] = [256, 13, 256, 3, 1]
# featmaps['input55'] = [256, 13, 256, 3, 1]
# featmaps['input2'] = [256, 27]
# featmaps['input3'] = [384, 13]
# featmaps['input4'] = [3, 227]
# featmaps['input5'] = [3, 227]
# featmaps['input6'] = [3, 227]
# featmaps['input7'] = [3, 227]
# featmaps['input8'] = [3, 227]
# featmaps['small'] = [512, 14]
# featmaps['med']   = [64, 112]
# featmaps['large'] = [256, 56]

batches  = [1]
kernels  = [3, 5, 7, 11]
num_outs = [96, 256, 384]
strides  = [1, 2, 4]

## CONF

def shcmd(cmd):
    subprocess.call(cmd, shell=True)

def shcom(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out = p.communicate()[0]
    return out

def main( args ):
    PLAT = args[1]
    THREADS=args[2]
    NETCONF='conv'
    NET=NETCONF + '.prototxt'
    OUTNAME=NETCONF + '-sweep.csv'
    OUTNAME1=NETCONF + '-fpops.csv'
    FINAL=NETCONF+'-'+PLAT+'-gflops.csv'
    
    shcom('rm -rf %s-%s*' % (NETCONF, PLAT))
    shcom('rm -rf %s-sweep.csv' % (NETCONF))
    shcom('rm -rf %s-fpops.csv' % (NETCONF))
    f = open(OUTNAME1, "wb")
    w = csv.writer(f)
    w.writerow(['layer','batch','channel','height','width','num_output','kernel_size','stride','out_dim','fpops'])
    
    for batch in batches:
        cmd = './change-dim.sh %s %s %s' % (NET, 1, batch)
        shcom(cmd)
        for k in featmaps:
            channel = featmaps[k][0]
            height  = featmaps[k][1]
            num_out = featmaps[k][2]
            kernel  = featmaps[k][3]
            stride  = featmaps[k][4]
            cmd = './change-dim.sh %s %s %s' % (NET, 2, channel)
            shcom(cmd)
            cmd = './change-dim.sh %s %s %s' % (NET, 3, height)
            shcom(cmd)
            cmd = './change-dim.sh %s %s %s' % (NET, 4, height)
            shcom(cmd)
            cmd = './change-entry.sh %s %s %s' % (NET, 'num_output', num_out)
            shcom(cmd)
            cmd = './change-entry.sh %s %s %s' % (NET, 'kernel_size', kernel)
            shcom(cmd)
            cmd = './change-entry.sh %s %s %s' % (NET, 'stride', stride)
            shcom(cmd)
            out_dim = float(height - kernel)/float(stride) + 1
            kernel_comp = pow(kernel,2) + pow(kernel,2) - 1
            fpops = ((kernel_comp * pow(out_dim, 2)) * channel + channel*pow(out_dim,2)) * num_out * batch
    
            w.writerow([NETCONF,batch,channel,height,height,num_out,kernel,stride,out_dim,fpops])
            if PLAT is 'cpu':
                cmd = 'OPENBLAS_NUM_THREADS=%s ./dummy --gpu 0 --network %s --layer_csv %s' % (THREADS, NET, OUTNAME)
            else:
                cmd = './dummy --gpu 1 --network %s --layer_csv %s' % (NET, OUTNAME)
            shcom(cmd)
    
    f.close()
    cmd ='sed "1s/^/layer,lat\\n/" %s > temp.txt' % (OUTNAME)
    shcom(cmd)
    shcom('mv temp.txt %s' % OUTNAME)
    f1 = file(OUTNAME, 'r')
    f2 = file(OUTNAME1, 'r')
    f3 = open(FINAL, "wb")
    w1 = csv.writer(f3)
    w1.writerow(['layer','batch','channel','height','width','num_output','kernel_size','stride','out_dim','fpops','lat','gflops'])
    
    c1 = csv.reader(f1)
    c2 = csv.reader(f2)
    
    next(c1, None)
    next(c2, None)
    
    for r1,r in zip(c1,c2):
        lat = float(r1[1])/1000
        gflops = float(r[9]) / lat / pow(10,9)
        w1.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r1[1],gflops])
    
    f3.close()

if __name__=='__main__':
    sys.exit(main(sys.argv))
