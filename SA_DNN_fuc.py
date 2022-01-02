import torch as torch
import torch.nn as nn
import time
import copy

relu = nn.ReLU(inplace=False)

#net1‘out generate dw-signals
def preparameters_to_signals(code,penalty_factor,b_list,trans_torch):
    outs=code*trans_torch
    dslow =outs[:,0].reshape(-1,1)
    dfast =outs[:,1].reshape(-1,1)
    dvfast=outs[:,2].reshape(-1,1)    
    fslow =outs[:,3].reshape(-1,1)
    ffast =outs[:,4].reshape(-1,1)
    fvfast=relu(1-fslow-ffast-penalty_factor)
    
    s=fslow*torch.exp(-b_list*dslow)+ffast*torch.exp(-b_list*dfast)+fvfast*torch.exp(-b_list*dvfast)
    pre_parameters=torch.cat((dslow,dfast,dvfast,fslow,ffast,fvfast), dim=1)
    return s,pre_parameters

#net2‘out generate dw-signals
def parameters_correct(correct_factor,pre_parameters,b_list):
    outs=correct_factor*pre_parameters
    dslow=outs[:,0].reshape(-1,1)
    dfast=outs[:,1].reshape(-1,1)
    dvfast=outs[:,2].reshape(-1,1)    
    pre_fslow=outs[:,3].reshape(-1,1)
    pre_ffast=outs[:,4].reshape(-1,1)
    pre_fvfast=outs[:,5].reshape(-1,1)
    
    ffast=pre_ffast/(pre_fslow+pre_ffast+pre_fvfast)
    fvfast=pre_fvfast/(pre_fslow+pre_ffast+pre_fvfast)
    fslow=1-ffast-fvfast
    
    s=fslow*torch.exp(-b_list*dslow)+ffast*torch.exp(-b_list*dfast)+fvfast*torch.exp(-b_list*dfast)
    parameters=torch.cat((dslow,dfast,dvfast,fslow,ffast,fvfast), dim=1)
    return s,parameters

#init net
def set_parameter_requires_grad(model, requires_grad=True):
    if requires_grad:
        for param in model.parameters():
            param.requires_grad = True
            
def initialize_model(model,requires_grad=True):
    model_ft = model  
    set_parameter_requires_grad(model_ft,requires_grad=True)
    return model_ft

#train model
def train_model(model1,model2,dataloaders,criterion,optimizer1,optimizer2,num_epochs,penalty_factor,model1_path,model2_path,patience,b_list,trans_torch):
    since = time.time()
    loss_history = [] 
    min_loss = 100.
    bad_epochs=0.
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        running_loss = 0.   
        model1.train()  
        model2.train()  
        for inputs in dataloaders:             
            
            with torch.autograd.set_grad_enabled(True):
                X=model1(inputs) 
                pre_outs,pre_parameters=preparameters_to_signals(X,penalty_factor,b_list,trans_torch) 
                correct_factor=model2(X)
                outs_fix,_=parameters_correct(correct_factor,pre_parameters,b_list)
                loss =criterion(pre_outs, inputs)+criterion(outs_fix, inputs)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            running_loss += loss.item()*inputs.size(0)
            
        epoch_loss = running_loss/len(dataloaders.dataset)

        print("Loss for updata: {} ,num_bad_epochs: {} ".format(epoch_loss,bad_epochs))
        loss_history.append(epoch_loss) 
        
        if epoch_loss<=min_loss:
            min_loss=epoch_loss
            best_model1 = copy.deepcopy(model1.state_dict())
            best_model2 = copy.deepcopy(model2.state_dict())
            bad_epochs = 0.
        else:
            bad_epochs += 1.
            if bad_epochs == patience:
                print("Early stop , loss for updata: {} ,num_bad_epochs: {} ".format(epoch_loss,bad_epochs))
                break
    
    model1.load_state_dict(best_model1)
    model2.load_state_dict(best_model2)
        
    torch.save(model1.state_dict(),model1_path)
    torch.save(model2.state_dict(),model2_path)
        
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    return model1,model2,loss_history

#test bi-exponal decay signals rate & Dvfast=0 if Fvfast==0 & return parameters
def dnn_test(organ_img,model1,model2,penalty_factor,b_list,trans_torch):
    code=model1(organ_img) 
    pre_outs,pre_parameters=preparameters_to_signals(code,penalty_factor,b_list,trans_torch) 
    code_fix=model2(code)
    outs_fix,organ_parameters=parameters_correct(code_fix,pre_parameters,b_list)    
    mask=(pre_parameters[:,5]!=0).float()
    organ_parameters[:,2]=(pre_parameters[:,2])*mask
    t2=(torch.sum(pre_parameters[:,5]==0)).float()
    t3=(torch.sum(pre_parameters[:,5]!=0)).float()
    
    mask2=torch.where(pre_parameters[:,5]==0)
    mask3=torch.where(pre_parameters[:,5]!=0)
    organ_parameters_bi=organ_parameters[mask2[0],:]
    organ_parameters_ti=organ_parameters[mask3[0],:]
    
    print(" there are {} % bi-exponal decay signals".format(t2/(t2+t3)*100))
    return outs_fix,organ_parameters_bi,organ_parameters_ti