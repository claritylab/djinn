#!/usr/bin/env python

import math
import subprocess, re, os, sys, csv

# featmaps = {}     #  c   h/w
featmaps['input'] = [64, 1]
featmaps['med']   = [256, 1]
featmaps['med']   = [384, 1]
featmaps['large'] = [446, 1]
featmaps['large2'] = [512, 1]
featmaps['input1'] = [576, 1]
featmaps['input2'] = [640, 1]
featmaps['input3'] = [768, 1]
featmaps['small2'] = [1000, 1]
featmaps['small3'] = [2000, 1]
featmaps['small4'] = [3000, 1]
featmaps['small5'] = [4000, 1]

# featmaps = [64, 128, 196, 256, 320, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3200, 4096]
batches  = [1, 64, 256, 320, 384, 446, 512, 576]
num_outs = [64, 128, 196, 256, 320, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 3200, 4096]

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
    NETCONF='fc'
    NET=NETCONF + '.prototxt'
    OUTNAME=NETCONF + '-sweep.csv'
    OUTNAME1=NETCONF + '-fpops.csv'
    FINAL=NETCONF+'-'+PLAT+'-gflops.csv'
    
    shcom('rm -rf %s-%s*' % (NETCONF, PLAT))
    shcom('rm -rf %s-sweep.csv' % (NETCONF))
    shcom('rm -rf %s-fpops.csv' % (NETCONF))
    f = open(OUTNAME1, "wb")
    w = csv.writer(f)
    w.writerow(['layer','batch','channel','height','width','num_output','fpops'])
    
    for batch in batches:
        cmd = './change-dim.sh %s %s %s' % (NET, 1, batch)
        shcom(cmd)
        for k in featmaps:
            # channel = featmaps[k][0]
            # height  = featmaps[k][1]
            channel = k
            height  = 1
            cmd = './change-dim.sh %s %s %s' % (NET, 2, channel)
            shcom(cmd)
            cmd = './change-dim.sh %s %s %s' % (NET, 3, height)
            shcom(cmd)
            cmd = './change-dim.sh %s %s %s' % (NET, 4, height)
            shcom(cmd)
            for num_out in num_outs:
                cmd = './change-entry.sh %s %s %s' % (NET, 'num_output', num_out)
                shcom(cmd)
                # calc FP Ops
                in_dim = height * height * channel
                out_dim = num_out
                fpops = in_dim * num_out * batch
    
                w.writerow([NETCONF,batch,channel,height,height,num_out,fpops])
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
    w1.writerow(['layer','batch','channel','height','width','num_output','fpops','lat','gflops'])
    
    c1 = csv.reader(f1)
    c2 = csv.reader(f2)
    
    next(c1, None)
    next(c2, None)
    
    for r1,r in zip(c1,c2):
        lat = float(r1[1])/1000
        gflops = float(r[6]) / lat / pow(10,9)
        w1.writerow([r[0],r[1],r[2],r[3],r[4],r[5],r[6],r1[1],gflops])
    
    f3.close()
if __name__=='__main__':
    sys.exit(main(sys.argv))
