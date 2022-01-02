# import libraries
import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

df=torch.load("D:\\ivim_pth\\parameters\\"+'df.pth')

a=np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,0])**2)))

#lsq-aic bi
lsq_bi_d0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,0])**2)))])

lsq_bi_d1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,1])**2)))])
                   
lsq_bi_d2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,2])**2)))])
                   
lsq_bi_f0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,3])**2)))])
                   
lsq_bi_f1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,4])**2)))])
                   
lsq_bi_f2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,5])**2)))])
                   
#lsq-aic tri
lsq_tri_d0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,0])**2)))])

lsq_tri_d1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,1])**2)))])
                   
lsq_tri_d2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,2])**2)))])
                   
lsq_tri_f0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,3])**2)))])
                   
lsq_tri_f1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,4])**2)))])
                   
lsq_tri_f2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='LSQ-AIC') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,5])**2)))])

#sa-dnn bi
dnn_bi_d0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,0])**2)))])

dnn_bi_d1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,1])**2)))])
                   
dnn_bi_d2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,2])**2)))])
                   
dnn_bi_f0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,3])**2)))])
                   
dnn_bi_f1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,4])**2)))])
                   
dnn_bi_f2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='bi')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='bi')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='bi')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='bi')].iloc[:,5])**2)))])
                   
#sa-dnn tri
dnn_tri_d0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,0])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,0])**2)))])

dnn_tri_d1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,1])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,1])**2)))])
                   
dnn_tri_d2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,2])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,2])**2)))])
                   
dnn_tri_f0=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,3])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,3])**2)))])
                   
dnn_tri_f1=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,4])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,4])**2)))])
                   
dnn_tri_f2=np.array([np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==100) & (df['decaytype']=='tri')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==50) & (df['decaytype']=='tri')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==33) & (df['decaytype']=='tri')].iloc[:,5])**2))),
                   np.sqrt(np.mean((np.array(df.loc[(df['algorithm']=='SA-DNN') & (df['SNR']==20) & (df['decaytype']=='tri')].iloc[:,5])**2)))])
                   
#plot bi-exponential decay signals' parameters std of sa-dnn and lsq-aic
X=np.array([100,50,33,20])
plt.figure(figsize=(20,10))
font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 20}

plt.subplot(2,3,1)
plt.gca().invert_xaxis()
plt.plot(X,lsq_bi_d0,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_bi_d0,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(1,8,2)/10000,font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplot(2,3,2)
plt.gca().invert_xaxis()
plt.plot(X,lsq_bi_d1,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_bi_d1,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,7,2)/100,font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplot(2,3,3)
plt.gca().invert_xaxis()
plt.plot(X,lsq_bi_d2,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_bi_d2,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,4,1)/10,font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplot(2,3,4)
plt.gca().invert_xaxis()
plt.plot(X,lsq_bi_f0,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_bi_f0,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,31,5),font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplot(2,3,5)
plt.gca().invert_xaxis()
plt.plot(X,lsq_bi_f1,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_bi_f1,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,49,12),font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplot(2,3,6)
plt.gca().invert_xaxis()
plt.plot(X,lsq_bi_f2,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_bi_f2,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,43,7),font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplots_adjust(wspace =0.28, hspace =0.28)
plt.savefig(r'D:\ivim_pth\png\simulation_rmse_curve_bi.png' , dpi = 800)


#plot tri-exponential decay signals' parameters std of sa-dnn and lsq-aic
X=np.array([100,50,33,20])
plt.figure(figsize=(20,10))
font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 20}

plt.subplot(2,3,1)
plt.gca().invert_xaxis()
plt.plot(X,lsq_tri_d0,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_tri_d0,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(1,11,3)/10000,font=font)
plt.legend(loc = 'upper left',prop=font)

plt.subplot(2,3,2)
plt.gca().invert_xaxis()
plt.plot(X,lsq_tri_d1,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_tri_d1,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(1,11,3)/100,font=font)
plt.legend(prop=font)

plt.subplot(2,3,3)
plt.gca().invert_xaxis()
plt.plot(X,lsq_tri_d2,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_tri_d2,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,5,1)/10,font=font)
plt.legend(prop=font)

plt.subplot(2,3,4)
plt.gca().invert_xaxis()
plt.plot(X,lsq_tri_f0,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_tri_f0,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(1,11,3),font=font)
plt.legend(prop=font)

plt.subplot(2,3,5)
plt.gca().invert_xaxis()
plt.plot(X,lsq_tri_f1,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_tri_f1,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,25,6),font=font)
plt.legend(prop=font)

plt.subplot(2,3,6)
plt.gca().invert_xaxis()
plt.plot(X,lsq_tri_f2,label="LSQ-AIC",linewidth=2.5)
plt.plot(X,dnn_tri_f2,label="SA-DNN",linewidth=2.5)
plt.xticks(np.arange(20,101,20),font=font)
plt.yticks(np.arange(0,25,6),font=font)
plt.legend(prop=font)

plt.subplots_adjust(wspace =0.28, hspace =0.28)
plt.savefig(r'D:\ivim_pth\png\simulation_rmse_curve_tri.png' , dpi = 800)
plt.show()
