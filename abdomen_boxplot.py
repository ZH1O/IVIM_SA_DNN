import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

df_abdominal_b=torch.load('D:\\ivim_pth\\'+'df_abdominal_b.pth')
df_abdominal_t=torch.load('D:\\ivim_pth\\'+'df_abdominal_t.pth')

font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 18}
plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
sns.boxplot(x="organ",y="Dslow",hue="algorithm",data=df_abdominal_b,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
#plt.ylim((0.0005,0.0015))
plt.ylabel("")
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,21,5)/10000,fontsize=20)
plt.legend(loc = 'upper right',fontsize=14)

plt.subplot(2,3,2)
sns.boxplot(x="organ",y="Dfast",hue="algorithm",data=df_abdominal_b,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylim((0.0,0.2))
plt.ylabel("")
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,21,5)/100,fontsize=20)
plt.legend(loc = 'upper right',fontsize=14)

plt.subplot(2,3,3)
sns.boxplot(x="organ",y="Dvfast",hue="algorithm",data=df_abdominal_b,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylabel("")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc = 'upper right',fontsize=14)

plt.subplot(2,3,4)
sns.boxplot(x="organ",y="Fslow",hue="algorithm",data=df_abdominal_b,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
#plt.ylim((65,80))
plt.ylabel("")
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,121,30),fontsize=20)
plt.legend(loc = 'upper right',fontsize=14)

plt.subplot(2,3,5)
sns.boxplot(x="organ",y="Ffast",hue="algorithm",data=df_abdominal_b,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
#plt.ylim((20,60))
plt.ylabel("")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yticks(np.arange(0,101,25),fontsize=20)
plt.legend(loc = 'upper right',fontsize=14)

plt.subplot(2,3,6)
sns.boxplot(x="organ",y="Fvfast",hue="algorithm",data=df_abdominal_b,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylabel("")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc = 'upper right',fontsize=14)

plt.subplots_adjust(wspace =0.32, hspace =0.2)
plt.savefig(r'D:\ivim_pth\png\abdominal_boxplot_bi.png' , dpi = 800)
plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
sns.boxplot(x="organ",y="Dslow",hue="algorithm",data=df_abdominal_t,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.legend(loc = 'upper right',fontsize=14)
#plt.ylim((0.0004,0.0014))
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,21,4)/10000,fontsize=20)
plt.ylabel("")

plt.subplot(2,3,2)
sns.boxplot(x="organ",y="Dfast",hue="algorithm",data=df_abdominal_t,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylim((0.00,0.1))
plt.legend(loc = 'upper right',fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,11,2)/100,fontsize=20)
plt.ylabel("")

plt.subplot(2,3,3)
sns.boxplot(x="organ",y="Dvfast",hue="algorithm",data=df_abdominal_t,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylim((0.,1.3))
plt.legend(loc = 'upper right',fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,131,30)/100,fontsize=20)
plt.ylabel("")

plt.subplot(2,3,4)
sns.boxplot(x="organ",y="Fslow",hue="algorithm",data=df_abdominal_t,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylim((0,120))
plt.legend(loc = 'upper right',fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,121,30),fontsize=20)
plt.ylabel("")

plt.subplot(2,3,5)
sns.boxplot(x="organ",y="Ffast",hue="algorithm",data=df_abdominal_t,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylim((0,80))
plt.legend(loc = 'upper right',fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,121,25),fontsize=20)
plt.ylabel("")

plt.subplot(2,3,6)
sns.boxplot(x="organ",y="Fvfast",hue="algorithm",data=df_abdominal_t,order=['Left liver lobe','Right liver lobe'])
plt.xlabel("")
plt.ylim((-5,60))
plt.legend(loc = 'upper right',fontsize=14)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,61,15),fontsize=20)
plt.ylabel("")

plt.subplots_adjust(wspace =0.32, hspace =0.2)
plt.savefig(r'D:\ivim_pth\png\abdominal_boxplot_tri.png' , dpi = 800)
plt.show()