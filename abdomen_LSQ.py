import numpy as np
import torch as torch
import os
from scipy.optimize import curve_fit
from LSQ_AIC_fuc import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]])
X=np.array([3,5,10,25,50,75,100,200,400,600,800])
_,b_num=b_list.shape

bi_parameters0=bi_parameters0_maker()
tri_parameters0=tri_parameters0_maker()


#all single file
namelist=['img_1l','img_2l','img_3l','img_4l','img_5l','img_6l','img_7l','img_8l','img_9l','img_10l',
          'img_1r','img_2r','img_3r','img_4r','img_5r','img_6r','img_7r','img_8r','img_9r','img_10r']

for test_name in namelist:
    #load data
    s=torch.load('D:\\ivim_pth\\'+test_name+'.pth').reshape(-1,12)[:,1:]
    print(s.shape)
    total_bi,total_tri=LSQ(s,X,bi_parameters0,tri_parameters0)
    
    torch.save(total_bi,'D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+test_name+'.pth')
    torch.save(total_tri,'D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+test_name+'.pth')

