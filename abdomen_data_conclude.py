import numpy as np
import torch as torch
import os
import torch.nn as nn
import pandas as pd
from SA_DNN_fuc import *
from model import *
from LSQ_AIC_fuc import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]])
_,b_num=b_list.shape

trans_torch=torch.tensor([[0.01,0.1,1.,1.,1.]])

img_1l=torch.load('D:\\ivim_pth\\'+r'img_1l.pth')
img_2l=torch.load('D:\\ivim_pth\\'+r'img_2l.pth')
img_3l=torch.load('D:\\ivim_pth\\'+r'img_3l.pth')
img_4l=torch.load('D:\\ivim_pth\\'+r'img_4l.pth')
img_5l=torch.load('D:\\ivim_pth\\'+r'img_5l.pth')
img_6l=torch.load('D:\\ivim_pth\\'+r'img_6l.pth')
img_7l=torch.load('D:\\ivim_pth\\'+r'img_7l.pth')
img_8l=torch.load('D:\\ivim_pth\\'+r'img_8l.pth')
img_9l=torch.load('D:\\ivim_pth\\'+r'img_9l.pth')
img_10l=torch.load('D:\\ivim_pth\\'+r'img_10l.pth')

img_1r=torch.load('D:\\ivim_pth\\'+r'img_1r.pth')
img_2r=torch.load('D:\\ivim_pth\\'+r'img_2r.pth')
img_3r=torch.load('D:\\ivim_pth\\'+r'img_3r.pth')
img_4r=torch.load('D:\\ivim_pth\\'+r'img_4r.pth')
img_5r=torch.load('D:\\ivim_pth\\'+r'img_5r.pth')
img_6r=torch.load('D:\\ivim_pth\\'+r'img_6r.pth')
img_7r=torch.load('D:\\ivim_pth\\'+r'img_7r.pth')
img_8r=torch.load('D:\\ivim_pth\\'+r'img_8r.pth')
img_9r=torch.load('D:\\ivim_pth\\'+r'img_9r.pth')
img_10r=torch.load('D:\\ivim_pth\\'+r'img_10r.pth')

pre_param_1l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_1l.pth")
pre_param_2l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_2l.pth")
pre_param_3l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_3l.pth")
pre_param_4l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_4l.pth")
pre_param_5l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_5l.pth")
pre_param_6l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_6l.pth")
pre_param_7l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_7l.pth")
pre_param_8l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_8l.pth")
pre_param_9l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_9l.pth")
pre_param_10l_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_10l.pth")

pre_param_1l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_1l.pth")
pre_param_2l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_2l.pth")
pre_param_3l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_3l.pth")
pre_param_4l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_4l.pth")
pre_param_5l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_5l.pth")
pre_param_6l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_6l.pth")
pre_param_7l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_7l.pth")
pre_param_8l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_8l.pth")
pre_param_9l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_9l.pth")
pre_param_10l_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_10l.pth")

pre_param_1r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_1r.pth")
pre_param_2r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_2r.pth")
pre_param_3r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_3r.pth")
pre_param_4r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_4r.pth")
pre_param_5r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_5r.pth")
pre_param_6r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_6r.pth")
pre_param_7r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_7r.pth")
pre_param_8r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_8r.pth")
pre_param_9r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_9r.pth")
pre_param_10r_bi=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_bi_"+"img_10r.pth")

pre_param_1r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_1r.pth")
pre_param_2r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_2r.pth")
pre_param_3r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_3r.pth")
pre_param_4r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_4r.pth")
pre_param_5r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_5r.pth")
pre_param_6r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_6r.pth")
pre_param_7r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_7r.pth")
pre_param_8r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_8r.pth")
pre_param_9r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_9r.pth")
pre_param_10r_tri=torch.load('D:\\ivim_pth\\parameters\\'+r"single_liver_tri_"+"img_10r.pth")

param_1l,param_1l_b,param_1l_t=AIC_select(pre_param_1l_bi,pre_param_1l_tri,img_1l,b_num)
param_2l,param_2l_b,param_2l_t=AIC_select(pre_param_2l_bi,pre_param_2l_tri,img_2l,b_num)
param_3l,param_3l_b,param_3l_t=AIC_select(pre_param_3l_bi,pre_param_3l_tri,img_3l,b_num)
param_4l,param_4l_b,param_4l_t=AIC_select(pre_param_4l_bi,pre_param_4l_tri,img_4l,b_num)
param_5l,param_5l_b,param_5l_t=AIC_select(pre_param_5l_bi,pre_param_5l_tri,img_5l,b_num)
param_6l,param_6l_b,param_6l_t=AIC_select(pre_param_6l_bi,pre_param_6l_tri,img_6l,b_num)
param_7l,param_7l_b,param_7l_t=AIC_select(pre_param_7l_bi,pre_param_7l_tri,img_7l,b_num)
param_8l,param_8l_b,param_8l_t=AIC_select(pre_param_8l_bi,pre_param_8l_tri,img_8l,b_num)
param_9l,param_9l_b,param_9l_t=AIC_select(pre_param_9l_bi,pre_param_9l_tri,img_9l,b_num)
param_10l,param_10l_b,param_10l_t=AIC_select(pre_param_10l_bi,pre_param_10l_tri,img_10l,b_num)
print("="*10)
param_1r,param_1r_b,param_1r_t=AIC_select(pre_param_1r_bi,pre_param_1r_tri,img_1r,b_num)
param_2r,param_2r_b,param_2r_t=AIC_select(pre_param_2r_bi,pre_param_2r_tri,img_2r,b_num)
param_3r,param_3r_b,param_3r_t=AIC_select(pre_param_3r_bi,pre_param_3r_tri,img_3r,b_num)
param_4r,param_4r_b,param_4r_t=AIC_select(pre_param_4r_bi,pre_param_4r_tri,img_4r,b_num)
param_5r,param_5r_b,param_5r_t=AIC_select(pre_param_5r_bi,pre_param_5r_tri,img_5r,b_num)
param_6r,param_6r_b,param_6r_t=AIC_select(pre_param_6r_bi,pre_param_6r_tri,img_6r,b_num)
param_7r,param_7r_b,param_7r_t=AIC_select(pre_param_7r_bi,pre_param_7r_tri,img_7r,b_num)
param_8r,param_8r_b,param_8r_t=AIC_select(pre_param_8r_bi,pre_param_8r_tri,img_8r,b_num)
param_9r,param_9r_b,param_9r_t=AIC_select(pre_param_9r_bi,pre_param_9r_tri,img_9r,b_num)
param_10r,param_10r_b,param_10r_t=AIC_select(pre_param_10r_bi,pre_param_10r_tri,img_10r,b_num)
print("="*10)

#save some data for curve fitting plot
torch.save(param_1r,'D:\\ivim_pth\\lsq_param_1r.pth')
torch.save(param_1l,'D:\\ivim_pth\\lsq_param_1l.pth')
torch.save(param_2r,'D:\\ivim_pth\\lsq_param_2r.pth')
torch.save(param_2l,'D:\\ivim_pth\\lsq_param_2l.pth')

param_r_b=torch.cat((param_1r_b,param_2r_b,param_3r_b,param_4r_b,param_5r_b,param_6r_b,param_7r_b,param_8r_b,param_9r_b,param_10r_b),dim=0).detach().numpy()
param_l_b=torch.cat((param_1l_b,param_2l_b,param_3l_b,param_4l_b,param_5l_b,param_6l_b,param_7l_b,param_8l_b,param_9l_b,param_10l_b),dim=0).detach().numpy()
param_r_t=torch.cat((param_1r_t,param_2r_t,param_3r_t,param_4r_t,param_5r_t,param_6r_t,param_7r_t,param_8r_t,param_9r_t,param_10r_t),dim=0).detach().numpy()
param_l_t=torch.cat((param_1l_t,param_2l_t,param_3l_t,param_4l_t,param_5l_t,param_6l_t,param_7l_t,param_8l_t,param_9l_t,param_10l_t),dim=0).detach().numpy()

clm=['Dslow','Dfast','Dvfast','Fslow','Ffast','Fvfast']

df_r_b=pd.DataFrame(param_r_b,columns=clm)
df_r_b.insert(6,'algorithm','LSQ-AIC')
df_r_b.insert(7,'organ','Left liver lobe')

df_l_b=pd.DataFrame(param_l_b,columns=clm)
df_l_b.insert(6,'algorithm','LSQ-AIC')
df_l_b.insert(7,'organ','Right liver lobe')

df_r_t=pd.DataFrame(param_r_t,columns=clm)
df_r_t.insert(6,'algorithm','LSQ-AIC')
df_r_t.insert(7,'organ','Left liver lobe')

df_l_t=pd.DataFrame(param_l_t,columns=clm)
df_l_t.insert(6,'algorithm','LSQ-AIC')
df_l_t.insert(7,'organ','Right liver lobe')

df_abdominal_lsq_b=df_r_b.append(df_l_b)

df_abdominal_lsq_t=df_r_t.append(df_l_t)

#SA-DNN
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

dnn_s_1l,param_1l_b,param_1l_t=dnn_test(img_1l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_2l,param_2l_b,param_2l_t=dnn_test(img_2l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_3l,param_3l_b,param_3l_t=dnn_test(img_3l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_4l,param_4l_b,param_4l_t=dnn_test(img_4l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_5l,param_5l_b,param_5l_t=dnn_test(img_5l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_6l,param_6l_b,param_6l_t=dnn_test(img_6l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_7l,param_7l_b,param_7l_t=dnn_test(img_7l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_8l,param_8l_b,param_8l_t=dnn_test(img_8l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_9l,param_9l_b,param_9l_t=dnn_test(img_9l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_10l,param_10_b,param_10l_t=dnn_test(img_10l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)

print("="*10)
dnn_s_1r,param_1r_b,param_1r_t=dnn_test(img_1r[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_2r,param_2r_b,param_2r_t=dnn_test(img_2r[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_3r,param_3r_b,param_3r_t=dnn_test(img_3r[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_4r,param_4r_b,param_4r_t=dnn_test(img_4r[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_5l,param_5l_b,param_5l_t=dnn_test(img_5l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_6l,param_6l_b,param_6l_t=dnn_test(img_6l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_7l,param_7l_b,param_7l_t=dnn_test(img_7l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_8l,param_8l_b,param_8l_t=dnn_test(img_8l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_9l,param_9l_b,param_9l_t=dnn_test(img_9l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)
dnn_s_10l,param_10l_b,param_10l_t=dnn_test(img_10l[:,:11],model1,model2,penalty_factor,b_list,trans_torch)

param_r_b=torch.cat((param_1r_b,param_2r_b,param_3r_b,param_4r_b,param_5r_b,param_6r_b,param_7r_b,param_8r_b,param_9r_b,param_10r_b),dim=0).detach().numpy()
param_l_b=torch.cat((param_1l_b,param_2l_b,param_3l_b,param_4l_b,param_5l_b,param_6l_b,param_7l_b,param_8l_b,param_9l_b,param_10l_b),dim=0).detach().numpy()
param_r_t=torch.cat((param_1r_t,param_2r_t,param_3r_t,param_4r_t,param_5r_t,param_6r_t,param_7r_t,param_8r_t,param_9r_t,param_10r_t),dim=0).detach().numpy()
param_l_t=torch.cat((param_1l_t,param_2l_t,param_3l_t,param_4l_t,param_5l_t,param_6l_t,param_7l_t,param_8l_t,param_9l_t,param_10l_t),dim=0).detach().numpy()


#save some data for curve fitting plot
torch.save(dnn_s_1r,'D:\\ivim_pth\\dnn_s_1r.pth')
torch.save(dnn_s_1l,'D:\\ivim_pth\\dnn_s_1l.pth')
torch.save(dnn_s_2r,'D:\\ivim_pth\\dnn_s_2r.pth')
torch.save(dnn_s_2l,'D:\\ivim_pth\\dnn_s_2l.pth')

clm=['Dslow','Dfast','Dvfast','Fslow','Ffast','Fvfast']

param_r_b[:,3:6]=param_r_b[:,3:6]*100
param_l_b[:,3:6]=param_l_b[:,3:6]*100
df_r_b=pd.DataFrame(param_r_b,columns=clm)
df_r_b.insert(6,'algorithm','SA-DNN')
df_r_b.insert(7,'organ','Right liver lobe')

df_l_b=pd.DataFrame(param_l_b,columns=clm)
df_l_b.insert(6,'algorithm','SA-DNN')
df_l_b.insert(7,'organ','Left liver lobe')

df_abdominal_dnn_b=df_r_b.append(df_l_b)

param_r_t[:,3:6]=param_r_t[:,3:6]*100
param_l_t[:,3:6]=param_l_t[:,3:6]*100
df_r_t=pd.DataFrame(param_r_t,columns=clm)
df_r_t.insert(6,'algorithm','SA-DNN')
df_r_t.insert(7,'organ','Right liver lobe')

df_l_t=pd.DataFrame(param_l_t,columns=clm)
df_l_t.insert(6,'algorithm','SA-DNN')
df_l_t.insert(7,'organ','Left liver lobe')

df_abdominal_dnn_t=df_r_t.append(df_l_t)

df_abdominal_b=df_abdominal_lsq_b.append(df_abdominal_dnn_b)
df_abdominal_t=df_abdominal_lsq_t.append(df_abdominal_dnn_t)

torch.save(df_abdominal_b,'D:\\ivim_pth\\'+'df_abdominal_b.pth')
torch.save(df_abdominal_t,'D:\\ivim_pth\\'+'df_abdominal_t.pth')
