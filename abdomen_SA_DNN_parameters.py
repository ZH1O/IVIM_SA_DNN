#generate IVIM parameters plot using SA-DNN

import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from SA_DNN_fuc import *
from model import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]])
_,b_num=b_list.shape

trans_torch=torch.tensor([[0.01,0.1,1.,1.,1.]])

relu = nn.ReLU(inplace=False)
noise_std=0.037
penalty_factor = noise_std*2.58
path0=r"D:/ivim_pth/model/ivim_"
model1_path=path0+r"_abdomennet1_"+str(penalty_factor)+r".pth"
model2_path=path0+r"_abdomennet2_"+str(penalty_factor)+r".pth"

model1 = initialize_model(model1(),requires_grad=False)
model2 = initialize_model(model2(),requires_grad=False)
model1.load_state_dict(torch.load(model1_path))
model2.load_state_dict(torch.load(model2_path))

all_pixels=torch.load('D:\\ivim_pth\\'+r"all_pixels.pth")

patient_num,layernum,length,_,b_num_pixels=all_pixels.shape

#generate one patient's imgs
all_pixels=all_pixels[0,0,:,:,:].reshape(-1,12)[:,:11]

code=model1(all_pixels)
_,pre_parameters=preparameters_to_signals(code,penalty_factor,b_list,trans_torch)
correct_factor=model2(code)
outs_fix,parameters=parameters_correct(correct_factor,pre_parameters,b_list)

mask=(parameters[:,5]!=0).float()
parameters[:,2]=(parameters[:,2])*mask

plt.rc('font',family='Times New Roman')
font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 15}

#save one patient's imgs

parameters=parameters.reshape(length,length,6)


plt.imshow((parameters[:,:,0]).cpu().detach().numpy(),cmap='gray',clim=(0.0005, 0.0025))
cb=plt.colorbar(pad=0.03,shrink=0.9)
cb.ax.tick_params(labelsize=14)
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\dnn_d0.png' ,dpi = 300)

plt.imshow((parameters[:,:,1]).cpu().detach().numpy(),cmap='gray',clim=(0.02, 0.1))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\dnn_d1.png' ,dpi = 300)

plt.imshow((parameters[:,:,2]).cpu().detach().numpy(),cmap='gray',clim=(0., 0.8))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\dnn_d2.png' ,dpi = 300)


plt.imshow((parameters[:,:,3]*100).cpu().detach().numpy(),cmap='gray',clim=(0, 100))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\dnn_f0.png' ,dpi = 300)


plt.imshow((parameters[:,:,4]*100).cpu().detach().numpy(),cmap='gray',clim=(0, 60))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\dnn_f1.png' ,dpi = 300)


plt.imshow((parameters[:,:,5]*100).cpu().detach().numpy(),cmap='gray',clim=(0, 40))
plt.xticks([])
plt.yticks([])
plt.savefig(r'D:\ivim_pth\png\dnn_f2.png' ,dpi = 300)