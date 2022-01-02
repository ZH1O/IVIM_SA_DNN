# import libraries
import numpy as np
import torch as torch
import time
import os
import copy
import torch.nn as nn
import torch.optim as optim
from SA_DNN_fuc import *
from model import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#set train parameters
#set parameters:"snr","noise_std" which contrl noise value 
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]]).to(device)
trans_torch=torch.tensor([[0.01,0.1,1.,1.,1.]]).to(device)

_,b_num=b_list.shape
noise_std=0.037
penalty_factor = noise_std*2.58
path0=r"D:/ivim_pth/model/ivim_"
model1_path=path0+r"_abdomennet1_"+str(penalty_factor)+r".pth"
model2_path=path0+r"_abdomennet2_"+str(penalty_factor)+r".pth"

#load train data
data=torch.load('D:\\ivim_pth\\'+r"organ_pixels.pth").to(device)
data=data[:,1:]

batch_size =128
dataloaders_dict = torch.utils.data.DataLoader(data,
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=0)

    
#train parameters
model1 = initialize_model(model1(),requires_grad=True)
model1=model1.to(device)

model2 = initialize_model(model2(),requires_grad=True)
modle2 = model2.to(device)


optimizer1 = optim.Adam(model1.parameters(), lr = 0.001)  
optimizer2 = optim.Adam(model2.parameters(), lr = 0.001)  

patience = 10
criterion = nn.MSELoss()  

#train
model1,model2,loss_history= train_model(model1,
                                        model2,
                                        dataloaders_dict,
                                        criterion, 
                                        optimizer1,
                                        optimizer2,
                                        num_epochs,
                                        penalty_factor,
                                        model1_path,
                                        model2_path,
                                        patience,
                                        b_list,
                                        trans_torch)