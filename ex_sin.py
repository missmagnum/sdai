import numpy as np
from pylab import *
import datetime

from gather_sda import Gather_sda
from knn import knn


def syn_ph(nsamp,nfeat,doplot=False):

    """    
    t = np.linspace(0,2*np.pi,100)
    uniform= sin(t+np.random.uniform(0,2*np.pi))
    plot(t,sin(t),'r',t,uniform,'b',t,uniform+np.random.normal(0,.5,100),'bo')
    """
    
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
         
    ### z_score mean 0 std 1     
    ###X_normalized=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    #### how about feature scaling ??     
    return X




    
#data_source = 'rna'
#dataset=loadtxt('rna_data.txt').T

dataset=syn_ph(1000,200)
print(dataset.shape)


percent = int(dataset.shape[0] * 0.8)   ### %80 of dataset for training
train, test_set = dataset[:percent] ,dataset[percent:]
percent_valid = int(train.shape[0] * 0.8)
train_set, valid_set = train[:percent_valid] , train[percent_valid:]


b_error=[]
mean_error=[]
knn_error=[]
missing_percent=np.linspace(0.1,0.9,9)
missing_percent=[0.1]

for mis in missing_percent:
    print('missing percentage: ',mis)

   
    available_mask=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)
    rest_mask, test_mask = available_mask[:percent], available_mask[percent:]
    train_mask = rest_mask[:percent_valid]
    valid_mask = rest_mask[percent_valid:]
    
    data= (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
   

    #### SDA
    
    gather=Gather_sda(dataset,data ,problem = 'class', available_mask = mask,
                      method = 'nes_mom',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0005,
                      training_epochs = 100,
                      finetune_lr = 0.0005,
                      batch_size = 100,
                      hidden_size = [200,20,2],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
      
    #knn_result = knn(dataset,available_mask)
    
    b_error.append(sum((1-available_mask)*((dataset-gather.gather_out())**2), axis=1).mean())
    mean_error.append(sum((1-available_mask)*((dataset-dataset.mean(axis=0))**2), axis=1).mean())
    #knn_error.append(sum((1-available_mask)*((dataset-knn_result)**2), axis=1).mean())
    plot(mis,b_error[-1],'ro')
    plot(mis,mean_error[-1],'bo')
    #plot(mis,knn_error[-1],'g*')
    
#plot(missing_percent,b_error,'r',missing_percent,knn_error,'g')
print(b_error)
#print(knn_error)
print(mean_error)
#show()
    
