# import libraries
import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#load data
data_bi_snr100=torch.load(r"D:/ivim_pth/data_bi_0.01.pth")
data_tri_snr100=torch.load(r"D:/ivim_pth/data_tri_0.01.pth")

data_bi_snr50=torch.load(r"D:/ivim_pth/data_bi_0.02.pth")
data_tri_snr50=torch.load(r"D:/ivim_pth/data_tri_0.02.pth")

data_bi_snr33=torch.load(r"D:/ivim_pth/data_bi_0.03.pth")
data_tri_snr33=torch.load(r"D:/ivim_pth/data_tri_0.03.pth")

data_bi_snr20=torch.load(r"D:/ivim_pth/data_bi_0.05.pth")
data_tri_snr20=torch.load(r"D:/ivim_pth/data_tri_0.05.pth")

lsq_signals_bi_snr100=np.load(r"D:/ivim_pth/parameters/lsq_signals_bi_snr100.npy")
lsq_signals_bi_snr50=np.load(r"D:/ivim_pth/parameters/lsq_signals_bi_snr50.npy")
lsq_signals_bi_snr33=np.load(r"D:/ivim_pth/parameters/lsq_signals_bi_snr33.npy")
lsq_signals_bi_snr20=np.load(r"D:/ivim_pth/parameters/lsq_signals_bi_snr20.npy")

lsq_signals_tri_snr100=np.load(r"D:/ivim_pth/parameters/lsq_signals_tri_snr100.npy")
lsq_signals_tri_snr50=np.load(r"D:/ivim_pth/parameters/lsq_signals_tri_snr50.npy")
lsq_signals_tri_snr33=np.load(r"D:/ivim_pth/parameters/lsq_signals_tri_snr33.npy")
lsq_signals_tri_snr20=np.load(r"D:/ivim_pth/parameters/lsq_signals_tri_snr20.npy")

dnn_signals_bi_snr100=np.load(r"D:/ivim_pth/parameters/dnn_signals_bi_snr100.npy")
dnn_signals_bi_snr50=np.load(r"D:/ivim_pth/parameters/dnn_signals_bi_snr50.npy")
dnn_signals_bi_snr33=np.load(r"D:/ivim_pth/parameters/dnn_signals_bi_snr33.npy")
dnn_signals_bi_snr20=np.load(r"D:/ivim_pth/parameters/dnn_signals_bi_snr20.npy")

dnn_signals_tri_snr100=np.load(r"D:/ivim_pth/parameters/dnn_signals_tri_snr100.npy")
dnn_signals_tri_snr50=np.load(r"D:/ivim_pth/parameters/dnn_signals_tri_snr50.npy")
dnn_signals_tri_snr33=np.load(r"D:/ivim_pth/parameters/dnn_signals_tri_snr33.npy")
dnn_signals_tri_snr20=np.load(r"D:/ivim_pth/parameters/dnn_signals_tri_snr20.npy")

#plot curve
plt.rc('font',family='Times New Roman')
b=np.array([0,3,5,10,25,50,75,100,200,400,600,800])
plt.figure(figsize=(20,10))

plt.subplot(2,4,1)
plt.scatter(b,data_bi_snr100[1100,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_bi_snr100[1100,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_bi_snr100[1100,:],label="SA-DNN",linewidth=2)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,2)
plt.scatter(b,data_bi_snr50[1000,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_bi_snr50[1000,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_bi_snr50[1000,:],label="SA-DNN",linewidth=2)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,3)
plt.scatter(b,data_bi_snr33[2900,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_bi_snr33[2900,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_bi_snr33[2900,:],label="SA-DNN",linewidth=2)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,4)
plt.scatter(b,data_bi_snr20[3000,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_bi_snr20[3000,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_bi_snr20[3000,:],label="SA-DNN",linewidth=2)

ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,5)
plt.scatter(b,data_tri_snr100[3000,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_tri_snr100[3000,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_tri_snr100[3000,:],label="SA-DNN",linewidth=2)

ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,6)
plt.scatter(b,data_tri_snr50[0,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_tri_snr50[0,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_tri_snr50[0,:],label="SA-DNN",linewidth=2)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,7)
plt.scatter(b,data_tri_snr33[3000,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_tri_snr33[3000,:],label="LSQ-AIC")
plt.plot(b,dnn_signals_tri_snr33[3000,:],label="SA-DNN",linewidth=2)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplot(2,4,8)
plt.scatter(b,data_tri_snr20[1800,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=4)
plt.plot(b,lsq_signals_tri_snr20[1800,:],label="LSQ-AIC",linewidth=2)
plt.plot(b,dnn_signals_tri_snr20[1800,:],label="SA-DNN",linewidth=2)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,820.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=18)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=18)
plt.legend(fontsize=18)

plt.subplots_adjust(wspace =0.4, hspace =0.3)
plt.savefig(r'D:\ivim_pth\png\\simulation_curveplot.png' , dpi = 800)
plt.show()