#generate dataframe & some files for plot.

import numpy as np
import torch as torch
import time
import os
import torch.nn as nn
import pandas as pd

from SA_DNN_fuc import *
from model import *
from LSQ_AIC_fuc import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#set parameters
b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]])
trans_torch=torch.tensor([[0.01,0.1,1.,1.,1.]])
_,b_num=b_list.shape
simulation_num=10000

#load model
path0=r"D:/ivim_pth/model/ivim_"

model1_snr100_path=path0+r"snr100"+r"_net1_"+str(0.01*2.58)+r".pth"
model2_snr100_path=path0+r"snr100"+r"_net2_"+str(0.01*2.58)+r".pth"

model1_snr50_path=path0+"snr50"+r"_net1_"+str(0.02*2.58)+r".pth"
model2_snr50_path=path0+"snr50"+r"_net2_"+str(0.02*2.58)+r".pth"

model1_snr33_path=path0+"snr33"+r"_net1_"+str(0.03*2.58)+r".pth"
model2_snr33_path=path0+"snr33"+r"_net2_"+str(0.03*2.58)+r".pth"

model1_snr20_path=path0+"snr20"+r"_net1_"+str(0.05*2.58)+r".pth"
model2_snr20_path=path0+"snr20"+r"_net2_"+str(0.05*2.58)+r".pth"

model1_snr100 = initialize_model(model1(),requires_grad=False)
model1_snr100.load_state_dict(torch.load(model1_snr100_path))
model2_snr100 = initialize_model(model2(),requires_grad=False)
model2_snr100.load_state_dict(torch.load(model2_snr100_path))

model1_snr50 = initialize_model(model1(),requires_grad=False)
model1_snr50.load_state_dict(torch.load(model1_snr50_path))
model2_snr50 = initialize_model(model2(),requires_grad=False)
model2_snr50.load_state_dict(torch.load(model2_snr50_path))

model1_snr33 = initialize_model(model1(),requires_grad=False)
model1_snr33.load_state_dict(torch.load(model1_snr33_path))
model2_snr33 = initialize_model(model2(),requires_grad=False)
model2_snr33.load_state_dict(torch.load(model2_snr33_path))

model1_snr20 = initialize_model(model1(),requires_grad=False)
model1_snr20.load_state_dict(torch.load(model1_snr20_path))
model2_snr20 = initialize_model(model2(),requires_grad=False)
model2_snr20.load_state_dict(torch.load(model2_snr20_path))

#load data
data_bi_snr100=torch.load(r"D:/ivim_pth/data_bi_0.01.pth")[:,1:]
data_tri_snr100=torch.load(r"D:/ivim_pth/data_tri_0.01.pth")[:,1:]

data_bi_snr50=torch.load(r"D:/ivim_pth/data_bi_0.02.pth")[:,1:]
data_tri_snr50=torch.load(r"D:/ivim_pth/data_tri_0.02.pth")[:,1:]

data_bi_snr33=torch.load(r"D:/ivim_pth/data_bi_0.03.pth")[:,1:]
data_tri_snr33=torch.load(r"D:/ivim_pth/data_tri_0.03.pth")[:,1:]

data_bi_snr20=torch.load(r"D:/ivim_pth/data_bi_0.05.pth")[:,1:]
data_tri_snr20=torch.load(r"D:/ivim_pth/data_tri_0.05.pth")[:,1:]


parameters_bi_snr100=torch.load(r"D:/ivim_pth/parameters_bi_0.01.pth")
parameters_tri_snr100=torch.load(r"D:/ivim_pth/parameters_tri_0.01.pth")

parameters_bi_snr50=torch.load(r"D:/ivim_pth/parameters_bi_0.02.pth")
parameters_tri_snr50=torch.load(r"D:/ivim_pth/parameters_tri_0.02.pth")

parameters_bi_snr33=torch.load(r"D:/ivim_pth/parameters_bi_0.03.pth")
parameters_tri_snr33=torch.load(r"D:/ivim_pth/parameters_tri_0.03.pth")

parameters_bi_snr20=torch.load(r"D:/ivim_pth/parameters_bi_0.05.pth")
parameters_tri_snr20=torch.load(r"D:/ivim_pth/parameters_tri_0.05.pth")

#record time for dnn test
since = time.time()

X_bi_snr100=model1_snr100(data_bi_snr100)
X_tri_snr100=model1_snr100(data_tri_snr100)
    
X_bi_snr50=model1_snr50(data_bi_snr50)
X_tri_snr50=model1_snr50(data_tri_snr50)
    
X_bi_snr33=model1_snr33(data_bi_snr33)
X_tri_snr33=model1_snr33(data_tri_snr33)

X_bi_snr20=model1_snr20(data_bi_snr20)
X_tri_snr20=model1_snr20(data_tri_snr20)
    
pre_outs_bi_snr100,pre_parameters_bi_snr100=preparameters_to_signals(X_bi_snr100,0.01*2.58,b_list,trans_torch)
pre_outs_tri_snr100,pre_parameters_tri_snr100=preparameters_to_signals(X_tri_snr100,0.01*2.58,b_list,trans_torch)

pre_outs_bi_snr50,pre_parameters_bi_snr50=preparameters_to_signals(X_bi_snr50,0.02*2.58,b_list,trans_torch)
pre_outs_tri_snr50,pre_parameters_tri_snr50=preparameters_to_signals(X_tri_snr50,0.02*2.58,b_list,trans_torch)

pre_outs_bi_snr33,pre_parameters_bi_snr33=preparameters_to_signals(X_bi_snr33,0.03*2.58,b_list,trans_torch)
pre_outs_tri_snr33,pre_parameters_tri_snr33=preparameters_to_signals(X_tri_snr33,0.03*2.58,b_list,trans_torch)

pre_outs_bi_snr20,pre_parameters_bi_snr20=preparameters_to_signals(X_bi_snr20,0.05*2.58,b_list,trans_torch)
pre_outs_tri_snr20,pre_parameters_tri_snr20=preparameters_to_signals(X_tri_snr20,0.05*2.58,b_list,trans_torch)


correction_factor_bi_snr100=model2_snr100(X_bi_snr100)
correction_factor_bi_snr50=model2_snr50(X_bi_snr50)
correction_factor_bi_snr33=model2_snr33(X_bi_snr33)
correction_factor_bi_snr20=model2_snr20(X_bi_snr20)

correction_factor_tri_snr100=model2_snr100(X_tri_snr100)
correction_factor_tri_snr50=model2_snr50(X_tri_snr50)
correction_factor_tri_snr33=model2_snr33(X_tri_snr33)
correction_factor_tri_snr20=model2_snr20(X_tri_snr20)


dnn_signals_bi_snr100,dnn_parameters_bi_snr100=parameters_correct(correction_factor_bi_snr100,pre_parameters_bi_snr100,b_list)
dnn_signals_bi_snr50,dnn_parameters_bi_snr50=parameters_correct(correction_factor_bi_snr50,pre_parameters_bi_snr50,b_list)
dnn_signals_bi_snr33,dnn_parameters_bi_snr33=parameters_correct(correction_factor_bi_snr33,pre_parameters_bi_snr33,b_list)
dnn_signals_bi_snr20,dnn_parameters_bi_snr20=parameters_correct(correction_factor_bi_snr20,pre_parameters_bi_snr20,b_list)

dnn_signals_tri_snr100,dnn_parameters_tri_snr100=parameters_correct(correction_factor_tri_snr100,pre_parameters_tri_snr100,b_list)
dnn_signals_tri_snr50,dnn_parameters_tri_snr50=parameters_correct(correction_factor_tri_snr50,pre_parameters_tri_snr50,b_list)
dnn_signals_tri_snr33,dnn_parameters_tri_snr33=parameters_correct(correction_factor_tri_snr33,pre_parameters_tri_snr33,b_list)
dnn_signals_tri_snr20,dnn_parameters_tri_snr20=parameters_correct(correction_factor_tri_snr20,pre_parameters_tri_snr20,b_list)

#Dvfast=0 if fvfast=0
mask_dnn_parameters_bi_snr100=dnn_parameters_bi_snr100[:,5]!=0
mask_dnn_parameters_bi_snr50=dnn_parameters_bi_snr50[:,5]!=0
mask_dnn_parameters_bi_snr33=dnn_parameters_bi_snr33[:,5]!=0
mask_dnn_parameters_bi_snr20=dnn_parameters_bi_snr20[:,5]!=0
mask_dnn_parameters_tri_snr100=dnn_parameters_tri_snr100[:,5]!=0
mask_dnn_parameters_tri_snr50=dnn_parameters_tri_snr50[:,5]!=0
mask_dnn_parameters_tri_snr33=dnn_parameters_tri_snr33[:,5]!=0
mask_dnn_parameters_tri_snr20=dnn_parameters_tri_snr20[:,5]!=0

dnn_parameters_bi_snr100[:,2]=dnn_parameters_bi_snr100[:,2]*mask_dnn_parameters_bi_snr100
dnn_parameters_bi_snr50[:,2]=dnn_parameters_bi_snr50[:,2]*mask_dnn_parameters_bi_snr50
dnn_parameters_bi_snr33[:,2]=dnn_parameters_bi_snr33[:,2]*mask_dnn_parameters_bi_snr33
dnn_parameters_bi_snr20[:,2]=dnn_parameters_bi_snr20[:,2]*mask_dnn_parameters_bi_snr20

dnn_parameters_tri_snr100[:,2]=dnn_parameters_tri_snr100[:,2]*mask_dnn_parameters_tri_snr100
dnn_parameters_tri_snr50[:,2]=dnn_parameters_tri_snr50[:,2]*mask_dnn_parameters_tri_snr50
dnn_parameters_tri_snr33[:,2]=dnn_parameters_tri_snr33[:,2]*mask_dnn_parameters_tri_snr33
dnn_parameters_tri_snr20[:,2]=dnn_parameters_tri_snr20[:,2]*mask_dnn_parameters_tri_snr20

time_elapsed = time.time() - since
print("SA-DNN compete in in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
print("="*10)

print('Accuracy of 4 snr bi_exponential signal (SA-DNN)')
print("accuracy: {}%".format(torch.sum(dnn_parameters_bi_snr100[:,5]==0)/simulation_num*100))
print("accuracy: {}%".format(torch.sum(dnn_parameters_bi_snr50[:,5]==0)/simulation_num*100))
print("accuracy: {}%".format(torch.sum(dnn_parameters_bi_snr33[:,5]==0)/simulation_num*100))
print("accuracy: {}%".format(torch.sum(dnn_parameters_bi_snr20[:,5]==0)/simulation_num*100))

print('Accuracy of 4 snr tri_exponential signal (SA-DNN)')
print("accuracy: {}%".format((torch.sum(dnn_parameters_tri_snr100[:,5]!=0)/simulation_num)*100))
print("accuracy: {}%".format((torch.sum(dnn_parameters_tri_snr50[:,5]!=0)/simulation_num)*100))
print("accuracy: {}%".format((torch.sum(dnn_parameters_tri_snr33[:,5]!=0)/simulation_num)*100))
print("accuracy: {}%".format((torch.sum(dnn_parameters_tri_snr20[:,5]!=0)/simulation_num)*100))


#calculate (predict parameters value - ture parameters value) for sa-dnn
dnn_df_bi_snr100=dnn_parameters_bi_snr100-parameters_bi_snr100
dnn_df_bi_snr50=dnn_parameters_bi_snr50-parameters_bi_snr50
dnn_df_bi_snr33=dnn_parameters_bi_snr33-parameters_bi_snr33
dnn_df_bi_snr20=dnn_parameters_bi_snr20-parameters_bi_snr20

dnn_df_tri_snr100=dnn_parameters_tri_snr100-parameters_tri_snr100
dnn_df_tri_snr50=dnn_parameters_tri_snr50-parameters_tri_snr50
dnn_df_tri_snr33=dnn_parameters_tri_snr33-parameters_tri_snr33
dnn_df_tri_snr20=dnn_parameters_tri_snr20-parameters_tri_snr20

#convert the unit of fslow,ffast,fvfast to percentage
dnn_df_bi_snr100[:,3:6]=dnn_df_bi_snr100[:,3:6]*100
dnn_df_bi_snr50[:,3:6]=dnn_df_bi_snr50[:,3:6]*100
dnn_df_bi_snr33[:,3:6]=dnn_df_bi_snr33[:,3:6]*100
dnn_df_bi_snr20[:,3:6]=dnn_df_bi_snr20[:,3:6]*100

dnn_df_tri_snr100[:,3:6]=dnn_df_tri_snr100[:,3:6]*100
dnn_df_tri_snr50[:,3:6]=dnn_df_tri_snr50[:,3:6]*100
dnn_df_tri_snr33[:,3:6]=dnn_df_tri_snr33[:,3:6]*100
dnn_df_tri_snr20[:,3:6]=dnn_df_tri_snr20[:,3:6]*100

#generate dataframe for sa-dnn
clm=['Dslow','Dfast','Dvfast','Fslow','Ffast','Fvfast']

dnn_df_bi_snr100=pd.DataFrame(dnn_df_bi_snr100.detach().numpy(),columns=clm)
dnn_df_bi_snr100.insert(6,'SNR',100)
dnn_df_bi_snr100.insert(7,'algorithm','SA-DNN')
dnn_df_bi_snr100.insert(8,'decaytype','bi')

dnn_df_bi_snr50=pd.DataFrame(dnn_df_bi_snr50.detach().numpy(),columns=clm)
dnn_df_bi_snr50.insert(6,'SNR',50)
dnn_df_bi_snr50.insert(7,'algorithm','SA-DNN')
dnn_df_bi_snr50.insert(8,'decaytype','bi')

dnn_df_bi_snr33=pd.DataFrame(dnn_df_bi_snr33.detach().numpy(),columns=clm)
dnn_df_bi_snr33.insert(6,'SNR',33)
dnn_df_bi_snr33.insert(7,'algorithm','SA-DNN')
dnn_df_bi_snr33.insert(8,'decaytype','bi')

dnn_df_bi_snr20=pd.DataFrame(dnn_df_bi_snr20.detach().numpy(),columns=clm)
dnn_df_bi_snr20.insert(6,'SNR',20)
dnn_df_bi_snr20.insert(7,'algorithm','SA-DNN')
dnn_df_bi_snr20.insert(8,'decaytype','bi')

dnn_df_tri_snr100=pd.DataFrame(dnn_df_tri_snr100.detach().numpy(),columns=clm)
dnn_df_tri_snr100.insert(6,'SNR',100)
dnn_df_tri_snr100.insert(7,'algorithm','SA-DNN')
dnn_df_tri_snr100.insert(8,'decaytype','tri')

dnn_df_tri_snr50=pd.DataFrame(dnn_df_tri_snr50.detach().numpy(),columns=clm)
dnn_df_tri_snr50.insert(6,'SNR',50)
dnn_df_tri_snr50.insert(7,'algorithm','SA-DNN')
dnn_df_tri_snr50.insert(8,'decaytype','tri')

dnn_df_tri_snr33=pd.DataFrame(dnn_df_tri_snr33.detach().numpy(),columns=clm)
dnn_df_tri_snr33.insert(6,'SNR',33)
dnn_df_tri_snr33.insert(7,'algorithm','SA-DNN')
dnn_df_tri_snr33.insert(8,'decaytype','tri')

dnn_df_tri_snr20=pd.DataFrame(dnn_df_tri_snr20.detach().numpy(),columns=clm)
dnn_df_tri_snr20.insert(6,'SNR',20)
dnn_df_tri_snr20.insert(7,'algorithm','SA-DNN')
dnn_df_tri_snr20.insert(8,'decaytype','tri')

dnn_df_bi=dnn_df_bi_snr100.append(dnn_df_bi_snr50).append(dnn_df_bi_snr33).append(dnn_df_bi_snr20)
dnn_df_tri=dnn_df_tri_snr100.append(dnn_df_tri_snr50).append(dnn_df_tri_snr33).append(dnn_df_tri_snr20)

#load lsq predict parameters
lsq_parameters_bi2bi_snr100=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2bi_snr100.pth")
lsq_parameters_bi2tri_snr100=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2tri_snr100.pth")
lsq_parameters_tri2bi_snr100=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2bi_snr100.pth")
lsq_parameters_tri2tri_snr100=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2tri_snr100.pth")

lsq_parameters_bi2bi_snr50=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2bi_snr50.pth")
lsq_parameters_bi2tri_snr50=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2tri_snr50.pth")
lsq_parameters_tri2bi_snr50=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2bi_snr50.pth")
lsq_parameters_tri2tri_snr50=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2tri_snr50.pth")

lsq_parameters_bi2bi_snr33=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2bi_snr33.pth")
lsq_parameters_bi2tri_snr33=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2tri_snr33.pth")
lsq_parameters_tri2bi_snr33=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2bi_snr33.pth")
lsq_parameters_tri2tri_snr33=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2tri_snr33.pth")

lsq_parameters_bi2bi_snr20=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2bi_snr20.pth")
lsq_parameters_bi2tri_snr20=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_bi2tri_snr20.pth")
lsq_parameters_tri2bi_snr20=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2bi_snr20.pth")
lsq_parameters_tri2tri_snr20=torch.load("D:\\ivim_pth\\parameters\\"+r"lsq_parameters_tri2tri_snr20.pth")



print('Accuracy of 4 snr bi_exponential signal (LSQ-AIC)')
lsq_parameters_bi_snr100=LSQ_AIC(lsq_parameters_bi2bi_snr100,lsq_parameters_bi2tri_snr100,data_bi_snr100,b_num,multi=2)
lsq_parameters_bi_snr50=LSQ_AIC(lsq_parameters_bi2bi_snr50,lsq_parameters_bi2tri_snr50,data_bi_snr50,b_num,multi=2)
lsq_parameters_bi_snr33=LSQ_AIC(lsq_parameters_bi2bi_snr33,lsq_parameters_bi2tri_snr33,data_bi_snr33,b_num,multi=2)
lsq_parameters_bi_snr20=LSQ_AIC(lsq_parameters_bi2bi_snr20,lsq_parameters_bi2tri_snr20,data_bi_snr20,b_num,multi=2)

print('Accuracy of 4 snr tri_exponential signal (LSQ-AIC)')
lsq_parameters_tri_snr100=LSQ_AIC(lsq_parameters_tri2bi_snr100,lsq_parameters_tri2tri_snr100,data_tri_snr100,b_num,multi=3)
lsq_parameters_tri_snr50=LSQ_AIC(lsq_parameters_tri2bi_snr50,lsq_parameters_tri2tri_snr50,data_tri_snr50,b_num,multi=3)
lsq_parameters_tri_snr33=LSQ_AIC(lsq_parameters_tri2bi_snr33,lsq_parameters_tri2tri_snr33,data_tri_snr33,b_num,multi=3)
lsq_parameters_tri_snr20=LSQ_AIC(lsq_parameters_tri2bi_snr20,lsq_parameters_tri2tri_snr20,data_tri_snr20,b_num,multi=3)

#calculate (predict parameters value - ture parameters value) for lsq
lsq_df_bi_snr100=lsq_parameters_bi_snr100-parameters_bi_snr100
lsq_df_bi_snr50=lsq_parameters_bi_snr50-parameters_bi_snr50
lsq_df_bi_snr33=lsq_parameters_bi_snr33-parameters_bi_snr33
lsq_df_bi_snr20=lsq_parameters_bi_snr20-parameters_bi_snr20

lsq_df_tri_snr100=lsq_parameters_tri_snr100-parameters_tri_snr100
lsq_df_tri_snr50=lsq_parameters_tri_snr50-parameters_tri_snr50
lsq_df_tri_snr33=lsq_parameters_tri_snr33-parameters_tri_snr33
lsq_df_tri_snr20=lsq_parameters_tri_snr20-parameters_tri_snr20

#convert the unit of fslow,ffast,fvfast to percentage
lsq_df_bi_snr100[:,3:6]=lsq_df_bi_snr100[:,3:6]*100
lsq_df_bi_snr50[:,3:6]=lsq_df_bi_snr50[:,3:6]*100
lsq_df_bi_snr33[:,3:6]=lsq_df_bi_snr33[:,3:6]*100
lsq_df_bi_snr20[:,3:6]=lsq_df_bi_snr20[:,3:6]*100

lsq_df_tri_snr100[:,3:6]=lsq_df_tri_snr100[:,3:6]*100
lsq_df_tri_snr50[:,3:6]=lsq_df_tri_snr50[:,3:6]*100
lsq_df_tri_snr33[:,3:6]=lsq_df_tri_snr33[:,3:6]*100
lsq_df_tri_snr20[:,3:6]=lsq_df_tri_snr20[:,3:6]*100

#generate dataframe for LSQ-AIC
clm=['Dslow','Dfast','Dvfast','Fslow','Ffast','Fvfast']

lsq_df_bi_snr100=pd.DataFrame(lsq_df_bi_snr100.numpy(),columns=clm)
lsq_df_bi_snr100.insert(6,'SNR',100)
lsq_df_bi_snr100.insert(7,'algorithm','LSQ-AIC')
lsq_df_bi_snr100.insert(8,'decaytype','bi')

lsq_df_bi_snr50=pd.DataFrame(lsq_df_bi_snr50.numpy(),columns=clm)
lsq_df_bi_snr50.insert(6,'SNR',50)
lsq_df_bi_snr50.insert(7,'algorithm','LSQ-AIC')
lsq_df_bi_snr50.insert(8,'decaytype','bi')

lsq_df_bi_snr33=pd.DataFrame(lsq_df_bi_snr33.numpy(),columns=clm)
lsq_df_bi_snr33.insert(6,'SNR',33)
lsq_df_bi_snr33.insert(7,'algorithm','LSQ-AIC')
lsq_df_bi_snr33.insert(8,'decaytype','bi')

lsq_df_bi_snr20=pd.DataFrame(lsq_df_bi_snr20.numpy(),columns=clm)
lsq_df_bi_snr20.insert(6,'SNR',20)
lsq_df_bi_snr20.insert(7,'algorithm','LSQ-AIC')
lsq_df_bi_snr20.insert(8,'decaytype','bi')

lsq_df_tri_snr100=pd.DataFrame(lsq_df_tri_snr100.numpy(),columns=clm)
lsq_df_tri_snr100.insert(6,'SNR',100)
lsq_df_tri_snr100.insert(7,'algorithm','LSQ-AIC')
lsq_df_tri_snr100.insert(8,'decaytype','tri')

lsq_df_tri_snr50=pd.DataFrame(lsq_df_tri_snr50.numpy(),columns=clm)
lsq_df_tri_snr50.insert(6,'SNR',50)
lsq_df_tri_snr50.insert(7,'algorithm','LSQ-AIC')
lsq_df_tri_snr50.insert(8,'decaytype','tri')

lsq_df_tri_snr33=pd.DataFrame(lsq_df_tri_snr33.numpy(),columns=clm)
lsq_df_tri_snr33.insert(6,'SNR',33)
lsq_df_tri_snr33.insert(7,'algorithm','LSQ-AIC')
lsq_df_tri_snr33.insert(8,'decaytype','tri')

lsq_df_tri_snr20=pd.DataFrame(lsq_df_tri_snr20.numpy(),columns=clm)
lsq_df_tri_snr20.insert(6,'SNR',20)
lsq_df_tri_snr20.insert(7,'algorithm','LSQ-AIC')
lsq_df_tri_snr20.insert(8,'decaytype','tri')

lsq_signals_bi_snr100=parameters_to_signals(lsq_parameters_bi_snr100.numpy())
lsq_signals_bi_snr50=parameters_to_signals(lsq_parameters_bi_snr50.numpy())
lsq_signals_bi_snr33=parameters_to_signals(lsq_parameters_bi_snr33.numpy())
lsq_signals_bi_snr20=parameters_to_signals(lsq_parameters_bi_snr20.numpy())
lsq_signals_tri_snr100=parameters_to_signals(lsq_parameters_tri_snr100.numpy())
lsq_signals_tri_snr50=parameters_to_signals(lsq_parameters_tri_snr50.numpy())
lsq_signals_tri_snr33=parameters_to_signals(lsq_parameters_tri_snr33.numpy())
lsq_signals_tri_snr20=parameters_to_signals(lsq_parameters_tri_snr20.numpy())

#Merge dataset
lsq_df_bi=lsq_df_bi_snr100.append(lsq_df_bi_snr50).append(lsq_df_bi_snr33).append(lsq_df_bi_snr20)
lsq_df_tri=lsq_df_tri_snr100.append(lsq_df_tri_snr50).append(lsq_df_tri_snr33).append(lsq_df_tri_snr20)

#add s0=0 into signals
s0=np.ones((10000,1))
lsq_signals_bi_snr100=np.concatenate((s0,lsq_signals_bi_snr100),axis=1)
lsq_signals_bi_snr50=np.concatenate((s0,lsq_signals_bi_snr50),axis=1)
lsq_signals_bi_snr33=np.concatenate((s0,lsq_signals_bi_snr33),axis=1)
lsq_signals_bi_snr20=np.concatenate((s0,lsq_signals_bi_snr20),axis=1)

lsq_signals_tri_snr100=np.concatenate((s0,lsq_signals_tri_snr100),axis=1)
lsq_signals_tri_snr50=np.concatenate((s0,lsq_signals_tri_snr50),axis=1)
lsq_signals_tri_snr33=np.concatenate((s0,lsq_signals_tri_snr33),axis=1)
lsq_signals_tri_snr20=np.concatenate((s0,lsq_signals_tri_snr20),axis=1)

dnn_signals_bi_snr100=np.concatenate((s0,dnn_signals_bi_snr100.detach().numpy()),axis=1)
dnn_signals_bi_snr50=np.concatenate((s0,dnn_signals_bi_snr50.detach().numpy()),axis=1)
dnn_signals_bi_snr33=np.concatenate((s0,dnn_signals_bi_snr33.detach().numpy()),axis=1)
dnn_signals_bi_snr20=np.concatenate((s0,dnn_signals_bi_snr20.detach().numpy()),axis=1)
dnn_signals_tri_snr100=np.concatenate((s0,dnn_signals_tri_snr100.detach().numpy()),axis=1)
dnn_signals_tri_snr50=np.concatenate((s0,dnn_signals_tri_snr50.detach().numpy()),axis=1)
dnn_signals_tri_snr33=np.concatenate((s0,dnn_signals_tri_snr33.detach().numpy()),axis=1)
dnn_signals_tri_snr20=np.concatenate((s0,dnn_signals_tri_snr20.detach().numpy()),axis=1)

#save predict signals
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_bi_snr100.npy",lsq_signals_bi_snr100)
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_bi_snr50.npy",lsq_signals_bi_snr50)
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_bi_snr33.npy",lsq_signals_bi_snr33)
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_bi_snr20.npy",lsq_signals_bi_snr20)

np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_tri_snr100.npy",lsq_signals_tri_snr100)
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_tri_snr50.npy",lsq_signals_tri_snr50)
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_tri_snr33.npy",lsq_signals_tri_snr33)
np.save("D:\\ivim_pth\\parameters\\"+r"lsq_signals_tri_snr20.npy",lsq_signals_tri_snr20)

np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_bi_snr100.npy",dnn_signals_bi_snr100)
np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_bi_snr50.npy",dnn_signals_bi_snr50)
np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_bi_snr33.npy",dnn_signals_bi_snr33)
np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_bi_snr20.npy",dnn_signals_bi_snr20)

np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_tri_snr100.npy",dnn_signals_tri_snr100)
np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_tri_snr50.npy",dnn_signals_tri_snr50)
np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_tri_snr33.npy",dnn_signals_tri_snr33)
np.save("D:\\ivim_pth\\parameters\\"+r"dnn_signals_tri_snr20.npy",dnn_signals_tri_snr20)

#save dataframe
df_bi=lsq_df_bi.append(dnn_df_bi)
df_tri=lsq_df_tri.append(dnn_df_tri)
df=df_bi.append(df_tri)
torch.save(df_bi,"D:\\ivim_pth\\parameters\\"+r"df_bi.pth")
torch.save(df_tri,"D:\\ivim_pth\\parameters\\"+r"df_tri.pth")
torch.save(df,"D:\\ivim_pth\\parameters\\"+r"df.pth")

print('='*10)
print('RMSE of 4 snr bi_exponential signal (LSQ-AIC)')
print(np.sqrt(np.mean(np.sum((lsq_signals_bi_snr100[:,1:]-data_bi_snr100.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((lsq_signals_bi_snr50[:,1:]-data_bi_snr50.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((lsq_signals_bi_snr33[:,1:]-data_bi_snr33.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((lsq_signals_bi_snr20[:,1:]-data_bi_snr20.numpy())**2,axis=1))))
print('RMSE of 4 snr bi_exponential signal (SA-DNN)')
print(np.sqrt(np.mean(np.sum((dnn_signals_bi_snr100[:,1:]-data_bi_snr100.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((dnn_signals_bi_snr50[:,1:]-data_bi_snr50.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((dnn_signals_bi_snr33[:,1:]-data_bi_snr33.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((dnn_signals_bi_snr20[:,1:]-data_bi_snr20.numpy())**2,axis=1))))
print('RMSE of 4 snr tri_exponential signal (LSQ-AIC)')
print(np.sqrt(np.mean(np.sum((lsq_signals_tri_snr100[:,1:]-data_tri_snr100.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((lsq_signals_tri_snr50[:,1:]-data_tri_snr50.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((lsq_signals_tri_snr33[:,1:]-data_tri_snr33.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((lsq_signals_tri_snr20[:,1:]-data_tri_snr20.numpy())**2,axis=1))))
print('RMSE of 4 snr tri_exponential signal (SA-DNN)')
print(np.sqrt(np.mean(np.sum((dnn_signals_tri_snr100[:,1:]-data_tri_snr100.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((dnn_signals_tri_snr50[:,1:]-data_tri_snr50.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((dnn_signals_tri_snr33[:,1:]-data_tri_snr33.numpy())**2,axis=1))))
print(np.sqrt(np.mean(np.sum((dnn_signals_tri_snr20[:,1:]-data_tri_snr20.numpy())**2,axis=1))))

print('='*10)
print('Std of 4 snr bi_exponential signal RMSE(LSQ-AIC)')
print(np.std(np.sum((lsq_signals_bi_snr100[:,1:]-data_bi_snr100.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((lsq_signals_bi_snr50[:,1:]-data_bi_snr50.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((lsq_signals_bi_snr33[:,1:]-data_bi_snr33.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((lsq_signals_bi_snr20[:,1:]-data_bi_snr20.numpy())**2,axis=1),ddof=1))
print('Std of 4 snr bi_exponential signal RMSE(SA-DNN)')
print(np.std(np.sum((dnn_signals_bi_snr100[:,1:]-data_bi_snr100.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((dnn_signals_bi_snr50[:,1:]-data_bi_snr50.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((dnn_signals_bi_snr33[:,1:]-data_bi_snr33.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((dnn_signals_bi_snr20[:,1:]-data_bi_snr20.numpy())**2,axis=1),ddof=1))
print('Std of 4 snr tri_exponential signal RMSE(LSQ-AIC)')
print(np.std(np.sum((lsq_signals_tri_snr100[:,1:]-data_tri_snr100.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((lsq_signals_tri_snr50[:,1:]-data_tri_snr50.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((lsq_signals_tri_snr33[:,1:]-data_tri_snr33.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((lsq_signals_tri_snr20[:,1:]-data_tri_snr20.numpy())**2,axis=1),ddof=1))
print('Std of 4 snr tri_exponential signal RMSE(SA-DNN)')
print(np.std(np.sum((dnn_signals_tri_snr100[:,1:]-data_tri_snr100.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((dnn_signals_tri_snr50[:,1:]-data_tri_snr50.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((dnn_signals_tri_snr33[:,1:]-data_tri_snr33.numpy())**2,axis=1),ddof=1))
print(np.std(np.sum((dnn_signals_tri_snr20[:,1:]-data_tri_snr20.numpy())**2,axis=1),ddof=1))


