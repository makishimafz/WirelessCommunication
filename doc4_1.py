import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import random


random.seed(0)


def QPSK(data):
    
    bit_num = len(data)

    # 入力データをBPSK変調
    data_BPSK = data*2-1

    # BPSKデータの奇数番目は実部に、偶数番目は虚部になるように合成
    data_BPSK = data_BPSK * 1 / np.sqrt(2)
    ich = data_BPSK[0:bit_num:2] 
    qch = data_BPSK[1:bit_num:2]

    return ich, qch



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
        #ic = np.arange(nsamp) + ic0 
        ic = np.arange(1, nsamp + 1) + ic0

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

ml_qpsk = 2

nloop = 10000
noe = 0
nod = 0
ramp2 = np.zeros(0)
theta2 = np.zeros(0)


for _ in range(nloop):
    data = np.random.rand(nd)>0.5

    #BPSK
    data_bpsk = data*2-1
    data_bpsk = np.array(data_bpsk)

    data_bpsk_q = np.zeros(len(data_bpsk))

    #BPSKフェージング
    iout,quot,ramp,rcos,rsin = sefade(data_bpsk, data_bpsk_q,itau,dlvl,th,n0,itnd1,n1,len(data_bpsk),tstp, fd,flat)

    ramp2 = np.hstack((ramp2,ramp))
    theta2 = np.hstack((theta2,np.angle(rcos+1j*rsin)))
    itnd1 = itnd1 + itnd0


fig = plt.figure(figsize=(6,6))
#weights1 = np.ones_like(ramp2) / len(ramp2)
#plt.hist(ramp2,bins=100,weights=weights1)
plt.hist(ramp2,bins=100,density=True)
plt.xlabel('amplitude')
plt.ylabel('density')
#plt.savefig("figure/hist_amp.png")

fig = plt.figure(figsize=(6,6))
#weights2 = np.ones_like(theta2) / len(theta2)
#plt.hist(theta2,bins=100,weights=weights2)
plt.hist(theta2,bins=100,density=True)
plt.xlabel('theta')
plt.ylabel('density')
#plt.savefig("figure/hist_theta.png")
#plt.show()