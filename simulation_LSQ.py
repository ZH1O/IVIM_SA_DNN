# import libraries
import numpy as np
import torch as torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from LSQ_AIC_fuc import *

#setting test parameters
#set parameters:"snr","noise_std" which contrl noise value 
b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]])
_,b_num=b_list.shape
X=np.array([3,5,10,25,50,75,100,200,400,600,800])#b_list numpy type
test_num=10000

bi_parameters0=bi_parameters0_maker()
tri_parameters0=tri_parameters0_maker()

snrlist=["snr100","snr50","snr33","snr20"]
noise_stdlist=[0.01,0.02,0.03,0.05]

for i in range(4):
    snr=snrlist[i]
    noise_std=noise_stdlist[i]
    #load data(std=0.02 ,snr=50)
    data_bi=torch.load(r"D:/ivim_pth/data_bi_"+str(noise_std)+".pth")[:,1:]
    data_tri=torch.load(r"D:/ivim_pth/data_tri_"+str(noise_std)+".pth")[:,1:]
    #test
    lsq_parameters_bi2bi,lsq_parameters_bi2tri=LSQ(data_bi,X,bi_parameters0,tri_parameters0)
    lsq_parameters_tri2bi,lsq_parameters_tri2tri=LSQ(data_tri,X,bi_parameters0,tri_parameters0)
    print(snr)
    #save predict parameters of lsq
    torch.save(lsq_parameters_bi2bi,"D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2bi_"+snr+r".pth")
    torch.save(lsq_parameters_bi2tri,"D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2tri_"+snr+r".pth")
    torch.save(lsq_parameters_tri2bi,"D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2bi_"+snr+r".pth")
    torch.save(lsq_parameters_tri2tri,"D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2tri_"+snr+r".pth")