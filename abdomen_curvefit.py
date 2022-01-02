import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

b=np.array([0,3,5,10,25,50,75,100,200,400,600,800])
plt.figure(figsize=(10,10))

font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 18}
plt.rc('font',family='Times New Roman')
s0=torch.ones(1)

#using for torch
def parameters_to_signals(outs):
    b=torch.Tensor([[0,3,5,10,25,50,75,100,200,400,600,800]])
    d0=outs[:,0].reshape(-1,1)
    d1=outs[:,1].reshape(-1,1)
    d2=outs[:,2].reshape(-1,1)    
    f0=outs[:,3].reshape(-1,1)
    f1=outs[:,4].reshape(-1,1)
    f2=outs[:,5].reshape(-1,1)
    s=f0*torch.exp(-b*d0)+f1*torch.exp(-b*d1)+f2*torch.exp(-b*d2)
    return s

plt.subplot(2,2,1)
plt.plot(b,parameters_to_signals(torch.load('D:\\ivim_pth\\'+'lsq_param_2l.pth'))[2,:].detach().numpy(),label="LSQ-AIC",linewidth=2.5)
plt.plot(b,torch.cat((s0,torch.load('D:\\ivim_pth\\'+'dnn_s_2l.pth')[2,:].detach())),label="SA-DNN",linewidth=2.5)
plt.scatter(b,torch.load('D:\\ivim_pth\\'+'img_2l.pth')[2,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=3)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,810.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=20)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=20)
plt.legend(fontsize=15,prop=font)

plt.subplot(2,2,2)
plt.plot(b,parameters_to_signals(torch.load('D:\\ivim_pth\\'+'lsq_param_2r.pth'))[9,:].detach().numpy(),label="LSQ-AIC",linewidth=2.5)
plt.plot(b,torch.cat((s0,torch.load('D:\\ivim_pth\\'+'dnn_s_2r.pth')[9,:].detach())),label="SA-DNN",linewidth=2.5)
plt.scatter(b,torch.load('D:\\ivim_pth\\'+'img_2r.pth')[9,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=3)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,810.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=20)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=20)
plt.legend(fontsize=15,prop=font)

plt.subplot(2,2,3)
plt.plot(b,parameters_to_signals(torch.load('D:\\ivim_pth\\'+'lsq_param_2l.pth'))[-6,:].detach().numpy(),label="LSQ-AIC",linewidth=2.5)
plt.plot(b,torch.cat((s0,torch.load('D:\\ivim_pth\\'+'dnn_s_2l.pth')[-6,:].detach())),label="SA-DNN",linewidth=2.5)
plt.scatter(b,torch.load('D:\\ivim_pth\\'+'img_2l.pth')[-6,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=3)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,810.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=20)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=20)
plt.legend(fontsize=15,prop=font)

plt.subplot(2,2,4)
plt.plot(b,parameters_to_signals(torch.load('D:\\ivim_pth\\'+'lsq_param_2r.pth'))[6,:].detach().numpy(),label="LSQ-AIC",linewidth=2.5)
plt.plot(b,torch.cat((s0,torch.load('D:\\ivim_pth\\'+'dnn_s_2r.pth')[6,:].detach())),label="SA-DNN",linewidth=2.5)
plt.scatter(b,torch.load('D:\\ivim_pth\\'+'img_2r.pth')[6,:],label="DW signals",linewidth=1.5,c='none',edgecolors='black',zorder=3)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim((-10.,810.))
plt.xticks(np.arange(0, 801, 200.0),fontsize=20)
plt.ylim((0.,1.05))
plt.yticks(np.arange(0, 101, 25)/100,fontsize=20)
plt.legend(fontsize=15,prop=font)

plt.subplots_adjust(wspace =0.3, hspace =0.25)
plt.savefig(r'D:\ivim_pth\png\abdominal_fitting_curve.png' , dpi = 800)
plt.show()