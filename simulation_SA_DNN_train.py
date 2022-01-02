# import libraries
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
relu = nn.ReLU(inplace=False)
test_num=10000
num_epochs = 100
patience = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
b_list=torch.tensor([[3,5,10,25,50,75,100,200,400,600,800]]).to(device)
trans_torch=torch.tensor([[0.01,0.1,1.,1.,1.]]).to(device)
_,b_num=b_list.shape
snrlist=["snr100","snr50","snr33","snr20"]
noise_stdlist=[0.01,0.02,0.03,0.05]

for i in range(4):
    snr=snrlist[i]
    noise_std=noise_stdlist[i]
    penalty_factor = noise_std*2.58
    
    path0=r"D:/ivim_pth/model/ivim_"
    model1_path=path0+snr+r"_net1_"+str(penalty_factor)+r".pth"
    model2_path=path0+snr+r"_net2_"+str(penalty_factor)+r".pth"
    
    #load train data
    data=torch.load(r"D:/ivim_pth/data_"+str(noise_std)+r".pth").to(device)
    data=data[:,1:]
    
    batch_size =128
    dataloaders_dict = torch.utils.data.DataLoader(data,
                                                    batch_size=batch_size,
                                                    shuffle=True, num_workers=0)
       
    #train parameters
    model1_ft = initialize_model(model1(),requires_grad=True)
    model1_ft=model1_ft.to(device)
    
    model2_ft = initialize_model(model2(),requires_grad=True)
    modle2_ft = model2_ft.to(device)
    
    optimizer1 = optim.Adam(model1_ft.parameters(), lr = 0.001)  
    optimizer2 = optim.Adam(model2_ft.parameters(), lr = 0.001)  
    
    criterion = nn.MSELoss()  
    #train
    model1_ft,model2_ft,loss_history= train_model(model1_ft,
                                            model2_ft,
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
    
    #load model for test
    
    model1_ft.load_state_dict(torch.load(model1_path))
    
    model2_ft.load_state_dict(torch.load(model2_path))
    
    #load test data and test
    #the data with noise std=0.02 (snr=100)
    
    data_bi=torch.load(r"D:/ivim_pth/data_bi_"+str(noise_std)+r".pth")[:,1:].to(device)
    data_tri=torch.load(r"D:/ivim_pth/data_tri_"+str(noise_std)+r".pth")[:,1:].to(device)
        
    X_bi=model1_ft(data_bi)
    X_tri=model1_ft(data_tri)
        
    _,pre_parameters_bi=preparameters_to_signals(X_bi,penalty_factor,b_list,trans_torch)
    _,pre_parameters_tri=preparameters_to_signals(X_tri,penalty_factor,b_list,trans_torch)
    
    #display the resust
    accuracy_bi=torch.sum(pre_parameters_bi[:,5]<=0.)
    accuracy_tri=test_num-torch.sum(pre_parameters_tri[:,5]<=0.)
    print(snr)
    print("there are {} % accuracy in bi_exponential decay signals".format(torch.true_divide(accuracy_bi*100,test_num)))
    print("there are {} % accuracy in tri_exponential decay signals".format(torch.true_divide(accuracy_tri*100,test_num)))
