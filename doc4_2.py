import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import random


random.seed(0)

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
            if round(sub2, 6) != 1:
                x1 = np.sin(np.pi*(1-alfs)/IPOINT*ym)/np.pi/(1-sub2)/ym/tstp
                x2 = np.cos(np.pi*(1+alfs)/IPOINT*ym)/np.pi*sub1/(1-sub2)
                xt = x1 + x2
            else:
                xt = alfs*tr*((1-2/np.pi)*np.cos(np.pi/4/alfs)+(1+2/np.pi)*np.sin(np.pi/4/alfs))/np.sqrt(2)
        if ncc==0:
            xh[i] = xt/IPOINT/tr
        elif ncc == 1:
            xh[i] = xt/tr
        else:
            print('error')
            break
    return xh


'''
def QPSK(data):
    
    bit_num = len(data)

    # 入力データをBPSK変調
    data_BPSK = data*2-1

    # BPSKデータの奇数番目は実部に、偶数番目は虚部になるように合成
    data_BPSK = data_BPSK * 1 / np.sqrt(2)
    ich = data_BPSK[0:bit_num:2] 
    qch = data_BPSK[1:bit_num:2]

    return ich, qch
    
'''

#fade.m
def fade(idata,qdata,nsamp,tstp,fd,no,counter,flat):
    if fd != 0:
        ac0 = np.sqrt(1/(2*(no+1)))
        as0 = np.sqrt(1/(2*no))
        ic0 = counter

        pai = np.pi
        wm = 2*pai*fd
        n = 4*no+2
        ts = tstp
        wmts = wm*ts
        paino = pai/no

        xc = np.zeros(nsamp)
        xs = np.zeros(nsamp)
        ic = np.arange(nsamp) + ic0 #　ここの修正

        for nn in range(1,no+1):
            cwn = np.cos(np.cos(2*pai*nn/n)*ic*wmts)
            xc = xc + np.cos(paino*nn)*cwn
            xs = xs + np.sin(paino*nn)*cwn
        
        cwmt = np.sqrt(2)*np.cos(ic*wmts)
        xc = (2*xc+cwmt)*ac0
        xs = 2*xs*as0

        ramp = np.sqrt(xc**2+xs**2)
        rcos = xc/ramp
        rsin = xs/ramp

        if flat == 1:
            iout = np.sqrt(xc**2+xs**2)*idata[0:nsamp] 
            qout = np.sqrt(xc**2+xs**2)*qdata[0:nsamp]

        else:
            iout = xc*idata[0:nsamp] - xs*qdata[0:nsamp]
            qout = xs*idata[0:nsamp] + xc*qdata[0:nsamp]
        
    else:
        iout = idata
        qout = qdata
    
    return iout,qout,ramp,rcos,rsin


#delay.m
def delay(idata,qdata,nsamp,idel):
    iout = np.zeros(nsamp)
    qout = np.zeros(nsamp)

    if idel != 0:
        iout[0:idel] = np.zeros(idel)
        qout[0:idel] = np.zeros(idel)
    
    iout[idel:nsamp] = idata[0:nsamp-idel]
    qout[idel:nsamp] = qdata[0:nsamp-idel]

    return iout, qout


#sefade.m
def sefade(idata,qdata,itau,dlvl,th,n0,itn,n1,nsamp,tstp, fd,flat):
    iout = np.zeros(nsamp)
    qout = np.zeros(nsamp)

    total_attn = sum(10**(-1*dlvl/10))

    for k in range(n1):
        atts = 10**(-0.05*int(dlvl[k]))
        if dlvl[k] == 40:
            atts = 0
        
        theta = th[k]*np.pi/180

        itmp,qtmp = delay(idata,qdata,nsamp,itau[k])
        itmp3,qtmp3,ramp,rcos,rsin = fade(itmp,qtmp,nsamp,tstp,fd,n0[k],itn[k],flat)

        iout = iout+atts*itmp3/np.sqrt(total_attn)
        qout = qout+atts*qtmp3/np.sqrt(total_attn)
    
    return iout,qout,ramp,rcos,rsin


tstp = 0.5*10**(-6)

sr = 1/tstp

itau = [0]

dlvl = np.array([0])

n0=[6]

th=[0.0]

itnd0=10000

itnd1 = np.array([1000])

n1 = 1

fd = 10

nd = 100

flat = 1


nloop = 10000
noe = 0
nod = 0

ml = 1      
br = sr*ml 
EBN0 = np.arange(0,21,1) 
IPOINT = 8
irfn=21
alfs=0.5

BER = 0
simu = []

xh = hrollfcoef(irfn, IPOINT,sr,alfs,1)
xh2 = hrollfcoef(irfn, IPOINT,sr,alfs,0)

for ebn0 in EBN0:
    BER = 0
    itnd1 = np.array([1000])
    for _ in range(nloop):
        data = np.random.rand(nd)>0.5
        data1 = data*2-1
        data1 =list(data1)

        oversample = [0]*IPOINT*nd
        data2 = oversample.copy()
        for j in range(len(data1)):
            data2[j*IPOINT] = data1[j]

        data3 = np.convolve(data2,xh)
        data3_qch = np.zeros(len(data3))

        spow = sum(data3*data3)/nd
        attn = spow*sr/br*10**(-ebn0/10)
        attn = np.sqrt(0.5*attn)
        
        inoise = np.random.randn(len(data3))*attn
        itnd1 = itnd1 + itnd0

        iout,quot,ramp,rcos,rsin = sefade(data3, data3_qch,itau,dlvl,th,n0,itnd1,n1,len(data3),tstp, fd,flat)

        data4 = iout + inoise
        

        data5  = np.convolve(data4, xh2)
        sampl = IPOINT*irfn
        data6 = data5[sampl:IPOINT*nd +sampl-1:IPOINT]
        demodata = data6>0

        noe2 = sum(abs(data^demodata))
        nod2 = len(data)

        BER += noe2/nod2

    print(BER/nloop)
    simu.append(BER/nloop)


y = special.erfc(np.sqrt(10**(EBN0/10)))/2

fig = plt.figure(figsize=(5,5))
plt.plot(EBN0,simu, label='Simulation',marker='.',markersize=12)
plt.plot(EBN0,y, label= 'Theory')
plt.legend(loc = 'upper right')
plt.xlim(0,20)
plt.ylim(0.00001, 1)
plt.semilogy()
plt.xlabel("S/N[dB]")
plt.ylabel("BER")
plt.grid(which='both')
#plt.savefig("figure/BER_fade.png")