import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import random


random.seed(0)

#パラメータ
para = 128
fftlen = 128
noc = 128
nd = 6

ml = 2
sr = 250000
br = sr*ml
gilen = 32
#ebn0 = 3
EBN0 = np.arange(0,11,1)
BER = []

nloop = 1000
noe = 0
nod = 0
eop = 0
nop = 0


def QPSKmod(paradata,para,nd,ml):

    for i in range(para):
        # 入力データをBPSK変調
        data_BPSK = paradata[i]*2-1

        # BPSKデータの奇数番目は実部に、偶数番目は虚部になるように合成
        ich_temp = data_BPSK[0:ml*nd:2]
        qch_temp = data_BPSK[1:ml*nd:2]
        if i == 0:
            ich = ich_temp
            qch = qch_temp
        else:
            ich = np.insert(ich,[j for j in range(i,nd*i+1,i)],ich_temp)
            qch = np.insert(qch,[j for j in range(i,nd*i+1,i)],qch_temp)

    return ich, qch

#ガードインターバル挿入
def giins(ich, qch, fftlen, gilen, nd):
    ich = ich.reshape(nd,fftlen)
    qch = qch.reshape(nd,fftlen)
    for g in range(nd):
        if g == 0:
            ich_out = np.r_[ich[g][-gilen:],ich[g]]
            qch_out = np.r_[qch[g][-gilen:],qch[g]]
        else:
            ich_temp = np.r_[ich[g][-gilen:],ich[g]]
            qch_temp = np.r_[qch[g][-gilen:],qch[g]]
    
            ich_out = np.append(ich_out, ich_temp)
            qch_out = np.append(qch_out, qch_temp)
    
    return ich_out, qch_out

def comb(ich, qch, attn):
        noise_i = np.random.randn(len(ich))*attn
        noise_q = np.random.randn(len(qch))*attn

        return ich + noise_i, qch + noise_q

#ガードインターバル除去
def girem(ich,qch,fftlen,gilen,nd):
    ich = ich.reshape(nd,fftlen)
    qch = qch.reshape(nd,fftlen)
    for g in range(nd):
        if g == 0:
            ich_out = ich[g][gilen:]
            qch_out = qch[g][gilen:]
        else:
            ich_temp = ich[g][gilen:]
            qch_temp = qch[g][gilen:]
    
            ich_out = np.append(ich_out, ich_temp)
            qch_out = np.append(qch_out, qch_temp)
    
    return ich_out, qch_out
    
#QPSK復調
def QPSKdemode(ich,qch,para,nd,ml):
    data = []
    if ml ==2:
        for i in range(para):
            for j in range(nd):
                data.append(ich[para*j+i]>0)
                data.append(qch[para*j+i]>0)
    return data


for ebn0 in EBN0:
    noe = 0
    nod = 0
    eop = 0
    nop = 0

    for iii in range(nloop):
        #データ生成
        seldata = np.random.rand(para*nd*ml)>0.5
        paradata = seldata.reshape(para,nd*ml)

        #QPSK変調
        ich, qch = QPSKmod(paradata,para,nd,ml)

        kmod = np.sqrt(2)
        ich1 = ich/kmod
        qch1 = qch/kmod

        #IFFT
        x = ich1 + 1j*qch1
        y = np.fft.ifft(x)
        ich2 = np.real(y)
        qch2 = np.imag(y)

        #ガードインターバル挿入
        ich3, qch3 = giins(ich2,qch2,fftlen,gilen,nd)
        fftlen2 = fftlen + gilen
  
        #雑音
        spow = sum(ich3*ich3 + qch3*qch3)/nd/para
        attn = spow*sr/br*10**(-ebn0/10)
        attn = np.sqrt(0.5*attn)

        #雑音追加
        ich4, qch4 = comb(ich3,qch3,attn)
        #ich4,qch4 = ich3,qch3 


        #ガードインターバル除去
        ich5, qch5 = girem(ich4,qch4,fftlen2,gilen,nd)
 
        #FFT
        rx = ich5 + 1j*qch5
        ry = np.fft.fft(rx)
        ich6 = np.real(ry)
        qch6 = np.imag(ry)

        ich7 = ich6*kmod
        qch7 = qch6*kmod

        #QPSK復調
        demodata = QPSKdemode(ich7,qch7,para, nd,ml)

        #BER
        noe2 = sum(abs(demodata^seldata))
        nod2 = len(seldata)

        noe = noe + noe2
        nod = nod + nod2

        if noe != 0:
            eop += 1
        else:
            eop = eop

        nop += 1
        #print(iii,noe2/nod2,eop)

    per = eop/nop
    ber = noe/nod
    BER.append(ber)
    print(ebn0,ber,per,nloop)

y_QPSK = special.erfc(np.sqrt(10**(EBN0/10)))/2
fig = plt.figure(figsize=(7,7))
plt.plot(EBN0,BER, label='QPSK(Simulation)',marker='.',markersize=12)
plt.plot(EBN0,y_QPSK, label= 'QPSK(Theory)')
plt.legend(loc = 'upper right')
plt.xlim(0,11)
plt.ylim(0.00001, 1)
plt.semilogy()
plt.xlabel("S/N[dB]")
plt.ylabel("BER")
plt.grid()
plt.savefig("figure/BER_OFDM.png")