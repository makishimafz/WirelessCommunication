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

#飛ばし値はランダムにすると良い
#itnd0_1=10000

#itnd0_2 = 15000

itnd0_11= random.randint(0, 100000)

itnd0_12 = random.randint(0, 100000)

itnd0_21= random.randint(0, 100000)

itnd0_22 = random.randint(0, 100000)

#itnd1_1 = np.array([1000])
#itnd1_2 = np.array([100000])

n1 = 1

fd = 10

nd = 100

flat = 1


nloop = 1000

ntantena = 2
nrantena = 2

noe = 0
nod = 0

ml = 1      
br = sr*ml 
EBN0 = np.arange(0,21,1) 
#EBN0 = np.array([16])
IPOINT = 8
irfn=21
alfs=0.5

BER = 0
simu = []

xh = hrollfcoef(irfn, IPOINT,sr,alfs,1)
xh2 = hrollfcoef(irfn, IPOINT,sr,alfs,0)

for ebn0 in EBN0:
    BER = 0
    itnd1_11 = np.array([1000])
    itnd1_12 = np.array([10000])

    itnd1_21 = np.array([2000])
    itnd1_22 = np.array([15000])

    for _ in range(nloop):
        #データ生成
        data_1 = np.random.rand(nd)>0.5
        data_2 = np.random.rand(nd)>0.5

        #BPSK
        data1_1 = data_1*2-1
        data1_1 =list(data1_1)

        data1_2 = data_2*2-1
        data1_2 =list(data1_2)

        #オーバーサンプリング
        oversample = [0]*IPOINT*nd

        data2_1 = oversample.copy()
        for j in range(len(data1_1)):
            data2_1[j*IPOINT] = data1_1[j]

        data2_2 = oversample.copy()
        for j in range(len(data1_2)):
            data2_2[j*IPOINT] = data1_2[j]

        data3_1 = np.convolve(data2_1,xh)
        data3_1_qch = np.zeros(len(data3_1))

        data3_2 = np.convolve(data2_2,xh)
        data3_2_qch = np.zeros(len(data3_2))

        spow_1 = sum(data3_1*data3_1)/nd
        attn_1 = spow_1*sr/br*10**(-ebn0/10)
        attn_1 = np.sqrt(0.5*attn_1)

        spow_2 = sum(data3_2*data3_2)/nd
        attn_2 = spow_2*sr/br*10**(-ebn0/10)
        attn_2 = np.sqrt(0.5*attn_2)


        inoise_11 = np.random.randn(len(data3_1))*attn_1
        inoise_21 = np.random.randn(len(data3_1))*attn_1

        inoise_12 = np.random.randn(len(data3_2))*attn_2
        inoise_22 = np.random.randn(len(data3_2))*attn_2

        itnd1_11 = itnd1_11 + itnd0_11
        itnd1_12 = itnd1_12 + itnd0_12
        itnd1_21 = itnd1_21 + itnd0_21
        itnd1_22 = itnd1_22 + itnd0_22

        iout11,qout11,ramp11,rcos11,rsin11 = sefade(data3_1/np.sqrt(nrantena), data3_1_qch/np.sqrt(nrantena),itau,dlvl,th,n0,itnd1_11,n1,len(data3_1),tstp, fd,flat)
        iout21,qout21,ramp21,rcos21,rsin21 = sefade(data3_1/np.sqrt(nrantena), data3_1_qch/np.sqrt(nrantena),itau,dlvl,th,n0,itnd1_21,n1,len(data3_1),tstp, fd,flat)

        iout12,qout12,ramp12,rcos12,rsin12 = sefade(data3_2/np.sqrt(nrantena), data3_2_qch/np.sqrt(nrantena),itau,dlvl,th,n0,itnd1_12,n1,len(data3_2),tstp, fd,flat)
        iout22,qout22,ramp22,rcos22,rsin22 = sefade(data3_2/np.sqrt(nrantena), data3_2_qch/np.sqrt(nrantena),itau,dlvl,th,n0,itnd1_22,n1,len(data3_2),tstp, fd,flat)

        h11 = ramp11
        h12 = ramp12
        h21 = ramp21
        h22 = ramp22

        W = []
        
        for i in range(len(h11)):
            mat = np.dot(np.matrix([[h11[i], h12[i]],[h21[i], h22[i]]]), np.matrix([[h11[i], h21[i]],[h12[i], h22[i]]])) + np.matrix([[inoise_11[i]*inoise_11[i], 0],[0, inoise_22[i]*inoise_22[i]]])
            mat_inv = np.linalg.inv(mat)
            W.append(np.dot(np.matrix([[h11[i], h21[i]],[h12[i], h22[i]]]),mat_inv))


        y1 = iout11 + iout12 + inoise_11 
        y2 = iout21 + iout22 + inoise_22

        data4_1 = []
        data4_2 = []
        for j in range(len(y1)):
            data4_1.append(float(np.dot(W[j],np.matrix([[y1[j]],[y2[j]]]))[0]))
            data4_2.append(float(np.dot(W[j],np.matrix([[y1[j]],[y2[j]]]))[1]))
        
    
        data5_1  = np.convolve(data4_1, xh2)
        data5_2  = np.convolve(data4_2, xh2)
        sampl = IPOINT*irfn
        data6_1 = data5_1[sampl:IPOINT*nd +sampl-1:IPOINT]
        data6_2 = data5_2[sampl:IPOINT*nd +sampl-1:IPOINT]
        demodata_1 = data6_1>0
        demodata_2 = data6_2>0

        noe = sum(abs(data_1^demodata_1)) + sum(abs(data_2^demodata_2)) 
        nod = len(data_1) + len(data_2)

        BER += noe/nod


    print(BER/nloop)
    simu.append(BER/nloop)

SISO = [0.13924299999999493,0.11980699999999606,0.10166599999999695,0.08520299999999699,0.07089599999999747,0.05833299999999839,0.047772999999999066,0.038825999999999423,
        0.03151799999999962,0.025140999999999965,0.020168000000000002,0.016281000000000028,0.012937000000000089,0.010387000000000039,0.008159000000000017,0.006580999999999999,
        0.005208999999999992,0.004093000000000007,0.0033270000000000044,0.002680000000000003,0.002076999999999997]

fig = plt.figure(figsize=(5,5))
plt.plot(EBN0,simu, label='MIMO (MMSE)',marker='.',markersize=12)
plt.plot(EBN0,SISO, label= 'SISO',marker='.',markersize=12)
plt.legend(loc = 'upper right')
plt.xlim(0,20)
plt.ylim(0.00001, 1)
plt.semilogy()
plt.xlabel("S/N[dB]")
plt.ylabel("BER")
plt.grid(which='both')
#plt.savefig("figure/BER_MIMO.png")
plt.show()