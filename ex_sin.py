import numpy as np
from pylab import *
import datetime

from gather_sda import Gather_sda
from knn import knn


def syn_ph(nsamp,nfeat,doplot=False):
    
    X = np.zeros((nsamp,nfeat))
    t = np.linspace(0,2*np.pi,nfeat)
    if doplot:
        figure(1)
        clf()
    for i in range(nsamp):
        ph = np.random.uniform(0,2*np.pi)
        X[i,:] = np.sin(t+ph) + np.random.normal(0,0.5,nfeat)
        if doplot:           
            plot(t,X[i,:],'r.')
    if doplot:
         plot(t,np.sin(t+ph),'b')         
 
    return X



dataset=syn_ph(1000,200)
print(dataset.shape)


percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
percent_valid = int(train.shape[0] * 0.8)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


b_error=[]
mean_error=[]
knn_error=[]
missing_percent=np.linspace(0.,0.9,10)
#missing_percent=[0.1]

for mis in missing_percent:
    print('missing percentage: ',mis)

   
    available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
    rest_mask, test_mask = available_mask[:percent], available_mask[percent:]
    train_mask = rest_mask[:percent_valid]
    valid_mask = rest_mask[percent_valid:]
    
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   

    #### SDA
    # method =  'rmsprop'  'adam'   'nes_mom'  'adadelta'  for this example 'adam' is the best
    gather=Gather_sda(dataset,data ,problem = 'regression', available_mask = mask,
                      method = 'adam',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0001,
                      training_epochs = 200,
                      finetune_lr = 0.0001,
                      batch_size = 100,
                      hidden_size = [100,20,2],
                      corruption_da = [0.1,  0.1, 0.1],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
      
    knn_result = knn(dataset,available_mask)
    
    b_error.append(sum((1-available_mask)*((dataset-gather.gather_out())**2), axis=1).mean())
    mean_error.append(sum((1-available_mask)*((dataset-dataset.mean(axis=0))**2), axis=1).mean())
    knn_error.append(sum((1-available_mask)*((dataset-knn_result)**2), axis=1).mean())
    plot(mis,b_error[-1],'ro')
    plot(mis,mean_error[-1],'bo')
    plot(mis,knn_error[-1],'g*')
    
plot(missing_percent,mean_error,'b',label='mean_row')
plot(missing_percent,knn_error,'g',label='knn' )
plot(missing_percent,b_error,'r',label='sda')
xlabel('corruption percentage')
ylabel('MSE')
title('dataset: shifted sin + noise')
legend(loc=4,prop={'size':9})
print(b_error)
print(knn_error)
print(mean_error)
show()
    
