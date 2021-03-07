import subprocess
import os
import os.path as osp

os.chdir("/home/ketang/ece408/libwb")

p1 = subprocess.Popen(["nvcc -std=c++11 -rdc=true -c mp1.cu -o mp1.o"], shell=True)
p1.wait(100)
p2 = subprocess.Popen(["nvcc -std=c++11 -o mp1 mp1.o lib/libwb.so"], shell=True)
p2.wait(100)

path = "/home/ketang/ece408/labs/data/mp1"
for i in os.walk(path):
    if (i[2] != []):
        t1 = osp.join(i[0], i[2][0])
        t2 = osp.join(i[0], i[2][1])
        t3 = osp.join(i[0], i[2][2])
        p3 = subprocess.Popen(["./mp1 -e %s -i %s,%s -o mp1myoutput -t vector" % (t3, t1,t2)], shell=True)
        p3.wait(100)
        #p4 = subprocess.Popen(["/bin/bash","./mp1myoutput"])
        #p4.wait(100)
