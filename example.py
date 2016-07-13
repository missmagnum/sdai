import numpy as np
from pylab import *


from gather_sda import gather_sda



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
    ### z_score mean 0 std 1     
    X_normalized=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    #### how about feature scaling ??     
    return X_normalized



   
dataset=syn_ph(1000,2000)

missing_percent = 0.6
corruption=np.random.binomial(n=1, p = 1-missing_percent, size = dataset.shape)
data_with_missing = dataset * corruption

sda=gather_sda(data_with_missing, missing_mask = 1-corruption , dA_initiall = False ,error_known = False )
#sda.pretraining()
sad,data=sda.finetuning()




"""
def mse(x,z):
    return list(((x - z) ** 2).mean(axis=0))
    
error=mse(data,sad.outout())
t = np.linspace(0,2*np.pi,nfeat)
plot(t,error)
        ##### regularization preformace parameters in network#####
        n_hidden_nodes
        learning_rate
        momentum
        weight decay parameters
        


"""
