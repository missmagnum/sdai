import numpy as np
from pylab import *
import datetime

from gather_sda import Gather_sda
from knn import knn

"""
Sample*features

"""
X=np.loadtxt('marray.txt')   # (102, 293)
dataset=(X-np.mean(X,axis=0))/np.std(X,axis=0)

print(dataset.shape)



percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
percent_valid = int(train.shape[0] * 0.8)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


b_error=[]
mean_error=[]
knn_error=[]
missing_percent=np.linspace(0.1,0.9,9)
sdaw=[]
#missing_percent=[0.9]



for mis in missing_percent:
    print('missing percentage: ',mis)

   
    available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
    rest_mask, test_mask = available_mask[:percent], available_mask[percent:]
    train_mask =  np.random.binomial(n=1, p = 1, size = train_set.shape)  ##rest_mask[:percent_valid]
    valid_mask = rest_mask[percent_valid:]
    
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   

    #### SDA
    # method =  'rmsprop'  'adam'   'nes_mom'  'adadelta'  
    gather=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'nes_mom',
                      pretraining_epochs = 30,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 5,
                      hidden_size = [400,100,20],
                      corruption_da = [0.1, 0.1, 0.1],
                      dA_initiall =  True,
                      error_known = True )
    
    gather.finetuning()
      
    knn_result = knn(dataset,available_mask)
    #########run the result for test
    dd_mask=test_mask
    dd = test_set
    
    b_error.append(sum((1-dd_mask)*((dd-gather.gather_out())**2), axis=1).mean())
    mean_error.append(sum((1-available_mask)*((dataset-dataset.mean(axis=0))**2), axis=1).mean())
    knn_error.append(sum((1-available_mask)*((dataset-knn_result)**2), axis=1).mean())
    plot(mis,b_error[-1],'ro')
    plot(mis,mean_error[-1],'bo')
    plot(mis,knn_error[-1],'g*')

    
    #### SDA with corruption in training
    train_mask =  rest_mask[:percent_valid]
        
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   
    gather=Gather_sda(dataset = test_set*test_mask,
                      portion_data = data,
                      problem = 'regression',
                      available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 10,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.00001,
                      batch_size = 5,
                      hidden_size = [300,200,100],
                      corruption_da = [0.1,  0.1, 0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()

    sdaw.append(sum((1-dd_mask)*((dd-gather.gather_out())**2), axis=1).mean())
    plot(mis,sdaw[-1],'m+')
    print(mis,b_error[-1],mean_error[-1],knn_error[-1],sdaw[-1])
    figtext(.02,.02, "problem = regression,available_mask = mask,method = 'nes_mom',pretraining_epochs = 30,pretrain_lr = 0.0001,training_epochs = 200,finetune_lr = 0.0001,batch_size = 5,hidden_size = [400,100,20],corruption_da = [0.1, 0.1, 0.1],dA_initiall =  True,error_known = True ")
    
plot(missing_percent,mean_error,'b',label='mean_row')
plot(missing_percent,knn_error,'g',label='knn' )
plot(missing_percent,b_error,'r',label='sda')
plot(missing_percent,sdaw,'m',label='sdaw')
xlabel('corruption percentage')
ylabel('MSE')
title('dataset: Ovarian Cancer Samples')  ###RNA
legend(loc=4,prop={'size':9})
show()

