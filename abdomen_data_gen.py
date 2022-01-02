import numpy as np
import torch as torch
import matplotlib.image as mpimg
import os
import pydicom as dicom
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def singel_pixel_loader(path):
    #load singel volume
    dcm = dicom.read_file(path)
    pixel=dcm.pixel_array.astype(np.float32())
    pixel=(torch.tensor(pixel))/255
    return pixel

def pixels_filter(in_torch):
    #select training signals
    out_torch=in_torch.reshape(-1,12)
    mask=torch.where(out_torch[:,0]!=0)
    out_torch=out_torch[mask[0],:]
    return out_torch

def pixels_average(pixel,step):
    #average in (1+step)**2 pixels
    n,l,d,_,h=pixel.shape
    out_torch=torch.empty((n,l,d-2*step,d-2*step,h))
    for x in range(d-2*step):
        for y in range(d-2*step):
            out_torch[:,:,x,y,:]=(torch.mean(pixel[:,:,x:x+2*step+1,y:y+2*step+1,:],dim=[2,3]))
    out_torch=out_torch/(out_torch[:,:,:,:,0].reshape(n,l,d-2*step,d-2*step,1)+0.0000001)
    return out_torch

def pixel_loader(path0,patient_num,multi_num,b_num,layermin,layermax,organ):
    #path0: rootpath of data
    #patient_num:the num of patient
    #multi_num:the duplication times
    #b_num:the num of b_values
    #layermin&layermax:the edge of layer need
    #organ:Hand drawn ROI mask
    img=torch.empty((patient_num,multi_num,layermax-layermin,352,352,b_num))
    organ_mask=torch.empty((patient_num,multi_num,layermax-layermin,350,350,1))
    for real in range(patient_num):
        for multi in range(multi_num):
            for layer in range(layermax-layermin):
                for x in range(b_num):
                    path1=str(real+1)
                    path2=r"/"+str(multi+1)+r"/"
                    path3=r"DICOM/"
                    path4=r"IM_"
                    num=((layer+layermin)*b_num)+(x+1)
                    path5=(4-len(str(num)))*"0"+str(num)
                    
                    path=path0+path1+path2+path3+path4+path5
                    img[real,multi,layer,:,:,x]=singel_pixel_loader(path)

                organ_mask_path=path0+path1+r"/"+organ+"/"+str(layermin+layer+1)+r".jpg"
                organ_mask[real,multi,layer,:,:,0]=torch.tensor(mpimg.imread(organ_mask_path).copy())[1:-1,1:-1]
    
    all_pixels=torch.mean(img,dim=1)
    average_pixels=pixels_average(all_pixels,step=1)
    
    organ_mask=(torch.mean(organ_mask,dim=1)!=0).float()
    organ_img=average_pixels*organ_mask
    organ_pixels=pixels_filter(organ_img)
    
    return average_pixels,organ_pixels

#'organ_pixels' are the signals under the mask ,using for train SA-DNN
#'img_1l'.etc, are the signals under the mask in one patient,using for compare SA-DNN and LSQ-AIC
#"average_pixels/all_pixels" are the all signals,using for generate IVIM parameters pictures

organ='liver'
#generate all pixels' signals
average_pixels,organ_pixels=pixel_loader(path0=r"D:/dataset/ivim/ivim/",patient_num=10,multi_num=5,b_num=12,layermin=0,layermax=5,organ=organ)


#genetate every singel patient‘s signals
_,img_1l=pixel_loader(path0=r"D:/dataset/ivim/single_test/1/",patient_num=1,multi_num=5,b_num=12,layermin=3,layermax=4,organ='mask1l')
_,img_1r=pixel_loader(path0=r"D:/dataset/ivim/single_test/1/",patient_num=1,multi_num=5,b_num=12,layermin=3,layermax=4,organ='mask1r')

_,img_2l=pixel_loader(path0=r"D:/dataset/ivim/single_test/2/",patient_num=1,multi_num=5,b_num=12,layermin=2,layermax=3,organ='mask2l')
_,img_2r=pixel_loader(path0=r"D:/dataset/ivim/single_test/2/",patient_num=1,multi_num=5,b_num=12,layermin=2,layermax=3,organ='mask2r')

_,img_3l=pixel_loader(path0=r"D:/dataset/ivim/single_test/3/",patient_num=1,multi_num=5,b_num=12,layermin=4,layermax=5,organ='mask3l')
_,img_3r=pixel_loader(path0=r"D:/dataset/ivim/single_test/3/",patient_num=1,multi_num=5,b_num=12,layermin=4,layermax=5,organ='mask3r')

_,img_4l=pixel_loader(path0=r"D:/dataset/ivim/single_test/4/",patient_num=1,multi_num=5,b_num=12,layermin=1,layermax=2,organ='mask4l')
_,img_4r=pixel_loader(path0=r"D:/dataset/ivim/single_test/4/",patient_num=1,multi_num=5,b_num=12,layermin=1,layermax=2,organ='mask4r')

_,img_5l=pixel_loader(path0=r"D:/dataset/ivim/single_test/5/",patient_num=1,multi_num=5,b_num=12,layermin=0,layermax=1,organ='mask5l')
_,img_5r=pixel_loader(path0=r"D:/dataset/ivim/single_test/5/",patient_num=1,multi_num=5,b_num=12,layermin=0,layermax=1,organ='mask5r')

_,img_6l=pixel_loader(path0=r"D:/dataset/ivim/single_test/6/",patient_num=1,multi_num=5,b_num=12,layermin=2,layermax=3,organ='mask6l')
_,img_6r=pixel_loader(path0=r"D:/dataset/ivim/single_test/6/",patient_num=1,multi_num=5,b_num=12,layermin=2,layermax=3,organ='mask6r')

_,img_7l=pixel_loader(path0=r"D:/dataset/ivim/single_test/7/",patient_num=1,multi_num=5,b_num=12,layermin=3,layermax=4,organ='mask7l')
_,img_7r=pixel_loader(path0=r"D:/dataset/ivim/single_test/7/",patient_num=1,multi_num=5,b_num=12,layermin=3,layermax=4,organ='mask7r')

_,img_8l=pixel_loader(path0=r"D:/dataset/ivim/single_test/8/",patient_num=1,multi_num=5,b_num=12,layermin=0,layermax=1,organ='mask8l')
_,img_8r=pixel_loader(path0=r"D:/dataset/ivim/single_test/8/",patient_num=1,multi_num=5,b_num=12,layermin=0,layermax=1,organ='mask8r')

_,img_9l=pixel_loader(path0=r"D:/dataset/ivim/single_test/9/",patient_num=1,multi_num=5,b_num=12,layermin=2,layermax=3,organ='mask9l')
_,img_9r=pixel_loader(path0=r"D:/dataset/ivim/single_test/9/",patient_num=1,multi_num=5,b_num=12,layermin=2,layermax=3,organ='mask9r')

_,img_10l=pixel_loader(path0=r"D:/dataset/ivim/single_test/10/",patient_num=1,multi_num=5,b_num=12,layermin=4,layermax=5,organ='mask10l')
_,img_10r=pixel_loader(path0=r"D:/dataset/ivim/single_test/10/",patient_num=1,multi_num=5,b_num=12,layermin=4,layermax=5,organ='mask10r')


#save pixels
torch.save(average_pixels,'D:\\ivim_pth\\'+r"all_pixels.pth")
torch.save(organ_pixels,'D:\\ivim_pth\\'+r"organ_pixels.pth")

torch.save(img_1l,'D:\\ivim_pth\\'+r"img_1l.pth")
torch.save(img_2l,'D:\\ivim_pth\\'+r"img_2l.pth")
torch.save(img_3l,'D:\\ivim_pth\\'+r"img_3l.pth")
torch.save(img_4l,'D:\\ivim_pth\\'+r"img_4l.pth")
torch.save(img_5l,'D:\\ivim_pth\\'+r"img_5l.pth")
torch.save(img_6l,'D:\\ivim_pth\\'+r"img_6l.pth")
torch.save(img_7l,'D:\\ivim_pth\\'+r"img_7l.pth")
torch.save(img_8l,'D:\\ivim_pth\\'+r"img_8l.pth")
torch.save(img_9l,'D:\\ivim_pth\\'+r"img_9l.pth")
torch.save(img_10l,'D:\\ivim_pth\\'+r"img_10l.pth")
torch.save(img_1r,'D:\\ivim_pth\\'+r"img_1r.pth")
torch.save(img_2r,'D:\\ivim_pth\\'+r"img_2r.pth")
torch.save(img_3r,'D:\\ivim_pth\\'+r"img_3r.pth")
torch.save(img_4r,'D:\\ivim_pth\\'+r"img_4r.pth")
torch.save(img_5r,'D:\\ivim_pth\\'+r"img_5r.pth")
torch.save(img_6r,'D:\\ivim_pth\\'+r"img_6r.pth")
torch.save(img_7r,'D:\\ivim_pth\\'+r"img_7r.pth")
torch.save(img_8r,'D:\\ivim_pth\\'+r"img_8r.pth")
torch.save(img_9r,'D:\\ivim_pth\\'+r"img_9r.pth")
torch.save(img_10r,'D:\\ivim_pth\\'+r"img_10r.pth")
