import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import random

random.seed(0)

#初期値
sr = 256000.0

ml_QPSK=2
ml_16QAM = 4
ml_64QAM = 6

IPOINT = 8
irfn=21
alfs=0.8


nd = 120
EBN0 = np.arange(0,20,1)

nloop = 1000

BER = 0

norm_QPSK = np.sqrt(2)
norm_16QAM = np.sqrt(10)
#norm_16QAM = 2*(2*np.sqrt(2)+np.sqrt(10))/4
norm_64QAM = np.sqrt(42)
#norm_64QAM = 2*(13*np.sqrt(2)+np.sqrt(10)+np.sqrt(26)+np.sqrt(58)+np.sqrt(34)+np.sqrt(74))/16

simu_QPSK = []
simu_16QAM = []
simu_64QAM = []

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

xh = hrollfcoef(irfn, IPOINT,sr,alfs,1)
xh2 = hrollfcoef(irfn, IPOINT,sr,alfs,0)


def Tran_QPSK(data):

    bit_num = len(data)

    # 入力データをBPSK変調
    data_BPSK = data*2-1

    # BPSKデータの奇数番目は実部に、偶数番目は虚部になるように合成
    data_BPSK = data_BPSK * 1 / norm_QPSK
    data_QPSK = data_BPSK[0:bit_num:2] + 1j * data_BPSK[1:bit_num:2]

    return data_QPSK


def Tran_16QAM(data):

    bit_num = len(data)
    sym_num = bit_num / 4

    # 入力bit系列をBPSK変調
    data_BPSK = data*2-1

    # 奇数番目のシンボル重みをもたせて合成
    temp1 = 2 * data_BPSK[0:bit_num:2] + data_BPSK[1:bit_num:2]

    # 出力結果のグレイ符号化
    temp2 = temp1.copy()
    temp1[temp2 == 3] = 1
    temp1[temp2 == 1] = 3

    data_16QAM = (temp1[0:int(sym_num * 2):2] + 1j * temp1[1:int(sym_num * 2):2]) * 1 / norm_16QAM

    return data_16QAM


def Tran_64QAM(data):
    
    bit_num = len(data)
    sym_num = bit_num / 4

    # 入力bit系列をBPSK変調
    data_BPSK = data*2-1

    # シンボル重みをもたせてで合成
    temp1 = 4 * data_BPSK[0:bit_num:3] + 2 * data_BPSK[1:bit_num:3] + data_BPSK[2:bit_num:3]

    # 出力結果のグレイ符号化
    temp2 = temp1.copy()
    temp1[temp2 == 7] = 3
    temp1[temp2 == 5] = 1
    temp1[temp2 == 3] = 5
    temp1[temp2 == 1] = 7
    temp1[temp2 == -3] = -1
    temp1[temp2 == -1] = -3

    data_64QAM = (temp1[0:int(sym_num * 3):2] + 1j * temp1[1:int(sym_num * 3):2]) * 1 / norm_64QAM

    return data_64QAM


def Rece(M,r,data):
    data_rec = []
    for i in data:
        thr = i.real
        thi = i.imag
        for cor in range(M):
            data_rec.append(thr>0)
            thr = r*2**(M-cor-1) - abs(thr)
        for coi in range(M):
            data_rec.append(thi>0)
            thi = r*2**(M-coi-1) - abs(thi)
    return data_rec


for ebn0 in EBN0:
    BER_QPSK = 0
    BER_16QAM = 0
    BER_64QAM = 0
    for _ in range(nloop):
        # 送信ビット列を生成
        data = np.random.rand(nd)>0.5

        # 各種変調
        data_QPSK = Tran_QPSK(data)
        data_16QAM = Tran_16QAM(data)
        data_64QAM = Tran_64QAM(data)

        oversample_QPSK = [0]*int(IPOINT*nd/ml_QPSK)
        oversample_16QAM = [0]*int(IPOINT*nd/ml_16QAM)
        oversample_64QAM = [0]*int(IPOINT*nd/ml_64QAM)


        #オーバーサンプリング
        data2_QPSK = oversample_QPSK.copy()
        for j in range(len(data_QPSK)):
            data2_QPSK[j*IPOINT] = data_QPSK[j]

        data2_16QAM = oversample_16QAM.copy()
        for j in range(len(data_16QAM)):
            data2_16QAM[j*IPOINT] = data_16QAM[j]

        data2_64QAM = oversample_64QAM.copy()
        for j in range(len(data_64QAM)):
            data2_64QAM[j*IPOINT] = data_64QAM[j]

        #送信フィルタ
        data3_r_QPSK = np.convolve(np.real(data2_QPSK), xh)
        data3_i_QPSK = np.convolve(np.imag(data2_QPSK), xh)

        data3_r_16QAM = np.convolve(np.real(data2_16QAM), xh)
        data3_i_16QAM = np.convolve(np.imag(data2_16QAM), xh)

        data3_r_64QAM = np.convolve(np.real(data2_64QAM), xh)
        data3_i_64QAM = np.convolve(np.imag(data2_64QAM), xh)


        #各種雑音
        #QPSK
        spow_QPSK = (sum(data3_r_QPSK*data3_r_QPSK) + sum(data3_i_QPSK*data3_i_QPSK))/nd*2


        attn_QPSK = spow_QPSK/int(ml_QPSK)*10**(-ebn0/10)
        attn_QPSK = np.sqrt(0.5*attn_QPSK)

        inoise_QPSK_r = np.random.randn(len(data3_r_QPSK))*attn_QPSK
        inoise_QPSK_i = np.random.randn(len(data3_i_QPSK))*attn_QPSK


        data4_r_QPSK = data3_r_QPSK + inoise_QPSK_r 
        data4_i_QPSK = data3_i_QPSK + inoise_QPSK_i


        #16QAM
    
        spow_16QAM = (sum(data3_r_16QAM*data3_r_16QAM)+sum(data3_i_16QAM*data3_i_16QAM))/nd*4
        attn_16QAM = spow_16QAM/int(ml_16QAM)*10**(-ebn0/10)
        attn_16QAM = np.sqrt(0.5*attn_16QAM)

        inoise_16QAM_r = np.random.randn(len(data3_r_16QAM))*attn_16QAM
        inoise_16QAM_i = np.random.randn(len(data3_i_16QAM))*attn_16QAM

        data4_r_16QAM = data3_r_16QAM + inoise_16QAM_r 
        data4_i_16QAM = data3_i_16QAM + inoise_16QAM_i

        #64QAM
        spow_64QAM = (sum(data3_r_64QAM*data3_r_64QAM) + sum(data3_i_64QAM*data3_i_64QAM))/nd*6
        attn_64QAM = spow_64QAM/int(ml_64QAM)*10**(-ebn0/10)
        attn_64QAM = np.sqrt(0.5*attn_64QAM)

        inoise_64QAM_r = np.random.randn(len(data3_r_64QAM))*attn_64QAM
        inoise_64QAM_i = np.random.randn(len(data3_i_64QAM))*attn_64QAM

        data4_r_64QAM = data3_r_64QAM + inoise_64QAM_r 
        data4_i_64QAM = data3_i_64QAM+ inoise_64QAM_i

        #受信フィルタ
        data5_r_QPSK  = np.convolve(data4_r_QPSK, xh2)
        data5_i_QPSK  = np.convolve(data4_i_QPSK, xh2)
        
        data5_r_16QAM  = np.convolve(data4_r_16QAM, xh2)
        data5_i_16QAM  = np.convolve(data4_i_16QAM, xh2)

        data5_r_64QAM  = np.convolve(data4_r_64QAM, xh2)
        data5_i_64QAM  = np.convolve(data4_i_64QAM, xh2)
        
        #同期
        sampl = 168
        data6_QPSK = data5_r_QPSK[sampl:IPOINT*int(nd/ml_QPSK) +sampl-1:IPOINT] +1j*data5_i_QPSK[sampl:IPOINT*int(nd/2) +sampl-1:IPOINT]
        data6_16QAM = data5_r_16QAM[sampl:IPOINT*int(nd/ml_16QAM) +sampl-1:IPOINT] +1j*data5_i_16QAM[sampl:IPOINT*int(nd/4) +sampl-1:IPOINT]
        data6_64QAM = data5_r_64QAM[sampl:IPOINT*int(nd/ml_64QAM) +sampl-1:IPOINT] +1j*data5_i_64QAM[sampl:IPOINT*int(nd/6) +sampl-1:IPOINT]


        #各種復調
        data7_QPSK = Rece(int(ml_QPSK/2),1/norm_QPSK,data6_QPSK)
        data7_16QAM = Rece(int(ml_16QAM/2),1/norm_16QAM,data6_16QAM)
        data7_64QAM = Rece(int(ml_64QAM/2),1/norm_64QAM,data6_64QAM)

        noe_QPSK = sum(abs(data^data7_QPSK))
        noe_16QAM = sum(abs(data^data7_16QAM))
        noe_64QAM = sum(abs(data^data7_64QAM))

        BER_QPSK += noe_QPSK/len(data)
        BER_16QAM += noe_16QAM/len(data)
        BER_64QAM += noe_64QAM/len(data)

    print(BER_QPSK/nloop, BER_16QAM/nloop, BER_64QAM/nloop)
    simu_QPSK.append(BER_QPSK/nloop)
    simu_16QAM.append(BER_16QAM/nloop)
    simu_64QAM.append(BER_64QAM/nloop)

#BER理論値
y_QPSK = special.erfc(np.sqrt(10**(EBN0/10)))/2

y_16QAM = 3*special.erfc(np.sqrt(10**(EBN0/10)*2/5))/8

y_64QAM = 7*special.erfc(np.sqrt(10**(EBN0/10)/7))/24

fig = plt.figure(figsize=(7,7))
plt.plot(EBN0,simu_QPSK, label='QPSK(Simulation)',marker='.',markersize=12)
plt.plot(EBN0,simu_16QAM, label='16QAM(Simulation)',marker='.',markersize=12)
plt.plot(EBN0,simu_64QAM, label='64QAM(Simulation)',marker='.',markersize=12)
plt.plot(EBN0,y_QPSK, label= 'QPSK(Theory)')
plt.plot(EBN0,y_16QAM, label= '16QAM(Theory)')
plt.plot(EBN0,y_64QAM, label= '64QAM(Theory)')
plt.legend(loc = 'upper right')
plt.xlim(0,20)
plt.ylim(0.00001, 1)
plt.semilogy()
plt.xlabel("S/N[dB]")
plt.ylabel("BER")
plt.grid()
plt.savefig("figure/BER2.png")