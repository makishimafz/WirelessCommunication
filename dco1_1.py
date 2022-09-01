import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import random

random.seed(0)

#初期値
sr = 256000.0
ml=1
br=sr*ml
nd = 100
EBN0 = np.arange(0,10,1)
IPOINT = 8
irfn=21
alfs=0.2

nloop = 1000
noe = 0
nod = 0

BER = 0

simu = []

#ロールオフフィルタ
'''
t=np.arange(-3,3,0.01)

T = 1
xh= 1/(np.pi*t*(1-(4*alfs*t/T)**2))*np.sin(2*np.pi*t*(1-alfs)/T) + 4*alfs/(np.pi*T*(1-(4*alfs*t/T)**2))*np.cos(2*np.pi*t*(1+alfs)/T)
xh2= 1/(np.pi*t*(1-(4*alfs*t/T)**2))*np.sin(2*np.pi*t*(1-alfs)/T) + 4*alfs/(np.pi*T*(1-(4*alfs*t/T)**2))*np.cos(2*np.pi*t*(1+alfs)/T)

fig = plt.figure(figsize=(4,5))
plt.plot(t,xh)
plt.xlim(0,1)
plt.xlabel("a")
plt.ylabel("BER")
plt.grid()
plt.show()
'''
#フィルタ関数
def hrollfcoef(irfn, IPOINT,sr,alfs,ncc):
    tr = sr
    tstp = 1/tr/IPOINT

    n = IPOINT*irfn
    xh=np.zeros(n)
    mid = (n/2)
    sub1 = 4*alfs*tr

    for i in range(n):
        icon = i -mid
        ym = icon
        
        if icon == 0:
            xt = (1-alfs+4*alfs/np.pi)*tr
        else:
            sub2 = 16*alfs*alfs*ym*ym/IPOINT/IPOINT
            if round(sub2,6) != 1:
                x1 = np.sin(np.pi*(1-alfs)/IPOINT*ym)/np.pi/(1-sub2)/ym/tstp
                x2 = np.cos(np.pi*(1+alfs)/IPOINT*ym)/np.pi*sub1/(1-sub2)
                xt = x1 + x2
            else:
                print(ym)
                xt = alfs*tr*((1-2/np.pi)*np.cos(np.pi/4/alfs)+(1+2/np.pi)*np.sin(np.pi/4/alfs))/np.sqrt(2)
        if ncc==0:
            xh[i] = xt/IPOINT/tr
        elif ncc == 1:
            xh[i] = xt/tr
        else:
            print('error')
            break
    return xh

xh = hrollfcoef(irfn, IPOINT,sr,alfs,1)
xh2 = hrollfcoef(irfn, IPOINT,sr,alfs,0)

fig = plt.figure(figsize=(4,5))
plt.plot(xh2)
plt.xlim(0,170)
plt.grid()
plt.savefig("figure/filter_rec.png")

fig = plt.figure(figsize=(4,5))
plt.plot(xh)
plt.xlim(0,170)
plt.grid()
plt.savefig("figure/filter_sen.png")
#plt.show()


data = np.random.rand(nd)>0.5

#1,-1の値
data1 = data*2-1
data1 =list(data1)

#BPSKデータの保存

fig = plt.figure(figsize=(5,5))
plt.plot(data1,marker='.',markersize=12)
plt.xlim(0,10)
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("figure/BPSK_data.png")

oversample = [0]*IPOINT*nd

#送信機作成
data2 = oversample.copy()
for j in range(len(data1)):
    data2[j*IPOINT] = data1[j]

#オーバーサンプリング時のデータを保存

fig = plt.figure(figsize=(5,5))
plt.plot(data2,marker='.',markersize=8)
plt.xlim(0,100)
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("figure/BPSK_samplingdata.png")

data3 = np.convolve(data2,xh)

fig = plt.figure(figsize=(5,5))
plt.plot(data3,marker='.',markersize=8)
plt.xlim(0,200)
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("figure/BPSK_signal_sen.png")


#受信機作成

data5  = np.convolve(data3, xh2)


fig = plt.figure(figsize=(5,5))
plt.plot(data5,marker='.',markersize=8)
plt.xlim(0,300)
plt.xlabel("Time[s]")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig("figure/BPSK_signal_rec.png")


sampl = 168
data6 = data5[sampl:IPOINT*nd +sampl-1:IPOINT]
demodata = data6>0
noe2 = sum(abs(data^demodata))
nod2 = len(data)
BER = noe2/nod2
print(BER)
#ビット誤り率