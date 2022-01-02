import torch as torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

test_num=10000
total_num = 50000
device = "cpu"
b_list=torch.tensor([[0,3,5,10,25,50,75,100,200,400,600,800]]).to(device)
_,b_num=b_list.shape

def f_init(size,limit,multi):
    #size:simulation number
    #limit:min value of fslow,ffast,fvfast
    #multi:bi_exponential or tri_exponential decay
    out = torch.rand(size,multi)
    out = out*(1-(multi)*limit)+(multi-1)*limit
    out = out/(torch.sum(out,dim = 1).reshape(size,1))
    return out

def signals_maker(size,limit,multi,noise_min,noise_max):
    #size:simulation number
    #limit:min value of fslow,ffast,fvfast
    #multi:bi_exponential or tri_exponential decay
    #noise_min&noise_max:set noise value
    dslow = torch.Tensor(size,1).uniform_(0.0005, 0.002).to(device)
    dfast = torch.Tensor(size,1).uniform_(0.01, 0.1).to(device)
    dvfast = torch.Tensor(size,1).uniform_(0.3, 0.5).to(device)
        
    f_torch=f_init(size,limit,multi).to(device)
    
    if multi==3:  
        fslow = f_torch[:,0].reshape(-1,1)
        ffast = f_torch[:,1].reshape(-1,1)  
        fvfast = f_torch[:,2].reshape(-1,1)     
    else:
        fslow = f_torch[:,0].reshape(-1,1)
        ffast = f_torch[:,1].reshape(-1,1)  
        fvfast = torch.zeros(size,1).to(device)
        dvfast = torch.zeros(size,1).to(device)
        
    s_no_noise = fslow*torch.exp(-b_list*dslow)+ffast*torch.exp(-b_list*dfast)+fvfast*torch.exp(-b_list*dvfast)
    noise1 = torch.normal(mean=0.0,std=torch.Tensor(size,1).uniform_(noise_min, noise_max)*torch.ones(1,b_num).float()).to(device)
    noise2 = torch.normal(mean=0.0,std=torch.Tensor(size,1).uniform_(noise_min, noise_max)*torch.ones(1,b_num).float()).to(device)

    s = s_no_noise+noise1
    s = torch.sqrt(s**2 + noise2**2)
    
    parm=torch.cat((dslow,dfast,dvfast,fslow,ffast,fvfast), dim=1)
    
    return s,s_no_noise,parm 

#generate and save data
data1,_,parameters1=signals_maker(total_num,limit=0.1,noise_min=0.01,noise_max=0.01,multi=2)
data2,_,parameters2=signals_maker(total_num,limit=0.1,noise_min=0.01,noise_max=0.01,multi=3)
data=torch.cat((data1,data2),dim=0)
torch.save(data,r"D:/ivim_pth/data_0.01.pth")#save train data (std=0.01,snr=100)
torch.save(data1[0:test_num,:],r"D:/ivim_pth/data_bi_0.01.pth")# save test data (std=0.01,snr=100)
torch.save(data2[0:test_num,:],r"D:/ivim_pth/data_tri_0.01.pth")# save test data (std=0.01,snr=100)
torch.save(parameters1[0:test_num,:],r"D:/ivim_pth/parameters_bi_0.01.pth")#save label
torch.save(parameters2[0:test_num,:],r"D:/ivim_pth/parameters_tri_0.01.pth")#save label


data1,_,parameters1=signals_maker(total_num,limit=0.1,noise_min=0.02,noise_max=0.02,multi=2)
data2,_,parameters2=signals_maker(total_num,limit=0.1,noise_min=0.02,noise_max=0.02,multi=3)
data=torch.cat((data1,data2),dim=0)
torch.save(data,r"D:/ivim_pth/data_0.02.pth")
torch.save(data1[0:test_num,:],r"D:/ivim_pth/data_bi_0.02.pth")
torch.save(data2[0:test_num,:],r"D:/ivim_pth/data_tri_0.02.pth")
torch.save(parameters1[0:test_num,:],r"D:/ivim_pth/parameters_bi_0.02.pth")
torch.save(parameters2[0:test_num,:],r"D:/ivim_pth/parameters_tri_0.02.pth")

data1,_,parameters1=signals_maker(total_num,limit=0.1,noise_min=0.03,noise_max=0.03,multi=2)
data2,_,parameters2=signals_maker(total_num,limit=0.1,noise_min=0.03,noise_max=0.03,multi=3)
data=torch.cat((data1,data2),dim=0)
torch.save(data,r"D:/ivim_pth/data_0.03.pth")
torch.save(data1[0:test_num,:],r"D:/ivim_pth/data_bi_0.03.pth")
torch.save(data2[0:test_num,:],r"D:/ivim_pth/data_tri_0.03.pth")
torch.save(parameters1[0:test_num,:],r"D:/ivim_pth/parameters_bi_0.03.pth")
torch.save(parameters2[0:test_num,:],r"D:/ivim_pth/parameters_tri_0.03.pth")

data1,_,parameters1=signals_maker(total_num,limit=0.1,noise_min=0.05,noise_max=0.05,multi=2)
data2,_,parameters2=signals_maker(total_num,limit=0.1,noise_min=0.05,noise_max=0.05,multi=3)
data=torch.cat((data1,data2),dim=0)
torch.save(data,r"D:/ivim_pth/data_0.05.pth")
torch.save(data1[0:test_num,:],r"D:/ivim_pth/data_bi_0.05.pth")
torch.save(data2[0:test_num,:],r"D:/ivim_pth/data_tri_0.05.pth")
torch.save(parameters1[0:test_num,:],r"D:/ivim_pth/parameters_bi_0.05.pth")
torch.save(parameters2[0:test_num,:],r"D:/ivim_pth/parameters_tri_0.05.pth")