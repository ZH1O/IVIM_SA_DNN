#generate IVIM parameters plot using LSQ-AIC

import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from SA_DNN_fuc import *
from model import *
from LSQ_AIC_fuc import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]])
X=np.array([3,5,10,25,50,75,100,200,400,600,800])
_,b_num=b_list.shape

all_pixels=torch.load('D:\\ivim_pth\\'+r"all_pixels.pth")
patient_num,layernum,length,_,b_num_pixels=all_pixels.shape

#generate one patient's imgs
all_pixels=all_pixels[0,0,:,:,:].reshape(-1,12)[:,:11]

bi_parameters0=bi_parameters0_maker()
tri_parameters0=tri_parameters0_maker()

total_bi,total_tri=LSQ(all_pixels,X,bi_parameters0,tri_parameters0)

s_bi=parameters_to_signals(total_bi)
s_tri=parameters_to_signals(total_tri)

total_bi=torch.from_numpy(total_bi)
total_tri=torch.from_numpy(total_tri)

mask_organ=(total_bi[:,0]!=0).float()

AIC_bi=AIC(s_bi,all_pixels.numpy(),3,b_num)
AIC_tri=AIC(s_tri,all_pixels.numpy(),5,b_num)

mask=torch.from_numpy(((AIC_bi-AIC_tri)<0).astype(np.int64))

total_params=torch.zeros(total_tri.shape)
total_params[:,0]=mask*total_bi[:,0]+(1-mask)*total_tri[:,0]
total_params[:,1]=mask*total_bi[:,1]+(1-mask)*total_tri[:,1]
total_params[:,2]=(1-mask)*total_tri[:,2]
total_params[:,3]=mask*total_bi[:,2]+(1-mask)*total_tri[:,3]
total_params[:,4]=mask*total_bi[:,3]+(1-mask)*total_tri[:,4]
total_params[:,5]=(1-mask)*total_tri[:,5]

plt.rc('font',family='Times New Roman')
font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 15}

#save one patient's imgs
parameters=total_params.reshape(length,length,6)


plt.imshow((parameters[:,:,0]).cpu().detach().numpy(),cmap='gray',clim=(0.0005, 0.0025))
cb=plt.colorbar(pad=0.03,shrink=0.9)
cb.ax.tick_params(labelsize=14)
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\lsq_d0.png' ,dpi = 300)

plt.imshow((parameters[:,:,1]).cpu().detach().numpy(),cmap='gray',clim=(0.02, 0.1))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\lsq_d1.png' ,dpi = 300)

plt.imshow((parameters[:,:,2]).cpu().detach().numpy(),cmap='gray',clim=(0., 0.8))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\lsq_d2.png' ,dpi = 300)


plt.imshow((parameters[:,:,3]*100).cpu().detach().numpy(),cmap='gray',clim=(0, 100))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\lsq_f0.png' ,dpi = 300)


plt.imshow((parameters[:,:,4]*100).cpu().detach().numpy(),cmap='gray',clim=(0, 60))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\lsq_f1.png' ,dpi = 300)


plt.imshow((parameters[:,:,5]*100).cpu().detach().numpy(),cmap='gray',clim=(0, 40))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\lsq_f2.png' ,dpi = 300)
