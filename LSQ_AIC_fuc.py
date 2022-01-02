import numpy as np
import torch
from scipy.optimize import curve_fit

#generate lsq initial parameters
def bi_parameters0_maker():
    f_torch=[]
    for x0 in range(50,225,25):
        for x1 in range(10,115,15):
            for x2 in range(1,11,2):
                dslow=x0/100000
                dfast=x1/1000
                fslow=x2/10
                f_torch.append([dslow,dfast,fslow])
    return f_torch

def tri_parameters0_maker():
    f_torch=[]
    for x0 in range(50,225,25):
        for x1 in range(10,115,15):
            for x2 in range(300,525,25):
                for x3 in range(1,11,2):
                    for x4 in range(1,11-x3,2):
                        dslow=x0/100000
                        dfast=x1/1000
                        dvfast=x2/1000
                        ffast=x3/10
                        fvfast=x4/10
                        f_torch.append([dslow,dfast,dvfast,ffast,fvfast])
    return f_torch

#use parameters generate signals
def parameters_to_signals(inputs):
    b=np.array([[3,5,10,25,50,75,100,200,400,600,800]])
    dslow=inputs[:,0].reshape(-1,1)
    dfast=inputs[:,1].reshape(-1,1)
    dvfast=inputs[:,2].reshape(-1,1)    
    fslow=inputs[:,3].reshape(-1,1)
    ffast=inputs[:,4].reshape(-1,1)
    fvfast=inputs[:,5].reshape(-1,1)
    signals=fslow*np.exp(-b*dslow)+ffast*np.exp(-b*dfast)+fvfast*np.exp(-b*dvfast)
    return signals

#lsq for bi-exponential decay model
def ivim_bi(b, dslow,dfast,fslow):
    return fslow*np.exp(-b*dslow) + (1-fslow)*np.exp(-b*dfast)


def order_bi(dslow,dfast,fslow):
    if dfast < dslow:
        dfast, dslow = dslow, dfast
        fslow = 1-fslow   
    return np.array([dslow,dfast,0,fslow,1-fslow,0])

def fit_least_squares_bi(X,Y,p0):
    try:
        bounds = (0, 1)
        params, _ = curve_fit(ivim_bi, X, Y, p0=p0, bounds=bounds)
        dslow,dfast,fslow = params[0], params[1], params[2]
        return order_bi(dslow,dfast,fslow)
    except:
        return np.array([0,0,0,0,0,0])
    
#lsq for tri-exponential decay model 

def ivim_tri(b, dslow,dfast,dvfast,ffast,fvfast):
    return (1-ffast-fvfast)*np.exp(-b*dslow) + ffast*np.exp(-b*dfast)+fvfast*np.exp(-b*dvfast)


def order_tri(dslow,dfast,dvfast,ffast,fvfast):
    params=np.array([[dslow,1-ffast-fvfast],[dfast,ffast],[dvfast,fvfast]])
    params=params[np.argsort(params[:,0]),:]
    dslow=params[0,0]
    dfast=params[1,0]
    dvfast=params[2,0]
    fslow=params[0,1]
    ffast=params[1,1]
    fvfast=params[2,1]
    return np.array([dslow,dfast,dvfast,fslow,ffast,fvfast])

def fit_least_squares_tri(X,Y,p0):
    try:
        bounds = (0, 1)
        params, _ = curve_fit(ivim_tri, X, Y, p0=p0, bounds=bounds)
        dslow,dfast,dvfast,ffast,fvfast = params[0], params[1], params[2], params[3], params[4],
        return order_tri(dslow,dfast,dvfast,ffast,fvfast)
    except:
        return np.array([0,0,0,0,0,0])
    
#Using LSQ generate bi-exponential decay model and tri-exponential decay model parameters
def LSQ(signals,X,bi_parameters0,tri_parameters0):
    x,y=signals.shape
    signals=np.array(signals)
    lsq_parameters_bi=np.zeros((x, 6))  
    lsq_parameters_tri=np.zeros((x, 6))    
    for i in range(x):
        Y=signals[i,:]
        loss2=100
        loss3=100
        for j in range(len(bi_parameters0)):       
            p=bi_parameters0[j]
            temp_parameters_bi=fit_least_squares_bi(X,Y,p)
            temp_parameters_bi=np.array(temp_parameters_bi).reshape(1,-1)
            signals_bi=parameters_to_signals(temp_parameters_bi)
            Y_bi=Y.reshape(1,-1)
            loss=np.mean((Y_bi-signals_bi)**2)
            if loss<loss2:
                loss2=loss
                parameters_bi=temp_parameters_bi
                
        for k in range(len(tri_parameters0)):
            p=tri_parameters0[k]
            temp_parameters_tri=fit_least_squares_tri(X,Y,p)
            temp_parameters_tri=np.array(temp_parameters_tri).reshape(1,-1)
            signals_tri=parameters_to_signals(temp_parameters_tri)
            Y_tri=Y.reshape(1,-1)
            loss=np.mean((Y_tri-signals_tri)**2)
            if loss<loss3:
                loss3=loss
                parameters_tri=temp_parameters_tri
        lsq_parameters_bi[i,:]=parameters_bi
        lsq_parameters_tri[i,:]=parameters_tri
    return lsq_parameters_bi,lsq_parameters_tri

#Akaike information criterion
def AIC(y_test, y_pred, k, n):
    resid =(y_test - y_pred)
    RSS = np.sum((resid ** 2),axis=1)
    AICValue = 2*k+n*np.log(RSS/n)
    return AICValue

#using AIC select lsq predict parameters
def LSQ_AIC(parameters_bi,parameters_tri,signals,b_num,multi):
    signals_bi=parameters_to_signals(parameters_bi)
    signals_tri=parameters_to_signals(parameters_tri)
    signals=signals.numpy()
    AIC_bi=AIC(signals_bi,signals,3,b_num)
    AIC_tri=AIC(signals_tri,signals,5,b_num)
    mask=torch.from_numpy(((AIC_bi-AIC_tri)<=0).astype(np.int64).reshape(-1,1))
    if multi==2:
        print("accuracy: {}%".format((torch.sum(mask)/mask.size(0))*100))
    else:
        print("accuracy: {}%".format((1-(torch.sum(mask)/mask.size(0)))*100))
    parameters=torch.from_numpy(parameters_bi)*mask+torch.from_numpy(parameters_tri)*(1-mask)  
    mask=parameters[:,5]!=0
    parameters[:,2]=parameters[:,2]*mask 
    return parameters

#using Akaike information criterion select parameters
def AIC_select(param2,param3,label,b_num):
    s2=parameters_to_signals(param2)
    s3=parameters_to_signals(param3)
    AIC_2bi=AIC(s2,label.numpy()[:,:11],3,b_num)
    AIC_2tri=AIC(s3,label.numpy()[:,:11],5,b_num)
    mask=torch.from_numpy(((AIC_2bi-AIC_2tri)<0).astype(np.int64).reshape(-1,1))
    num=torch.sum(mask)
    total_num,_=s2.shape
    print(" there are {} % bi-exponal decay signals".format((num/total_num)*100))
    param=torch.from_numpy(param2)*mask+torch.from_numpy(param3)*(1-mask)  
    mask=param[:,5]!=0
    param[:,2]=param[:,2]*mask 
    
    mask2=torch.where(param[:,5]==0)
    mask3=torch.where(param[:,5]!=0)
    organ_parameters_bi=torch.tensor(np.median(param[mask2[0],:].detach().numpy(),axis=0,keepdims=True))
    organ_parameters_ti=torch.tensor(np.median(param[mask3[0],:].detach().numpy(),axis=0,keepdims=True))
    
    return param,organ_parameters_bi,organ_parameters_ti