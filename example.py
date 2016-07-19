import numpy as np
from pylab import *
import datetime

from gather_sda import Gather_sda
from knnimpute import Knn


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


def syn_multid(nsamp,nfeat):
    t = np.linspace(0.,3.,nsamp)
    funcs=[lambda x: x**2, lambda x : log(x), lambda x: x**3, lambda x: x,lambda x: sin(x),lambda x: tan(x)]
    X = np.zeros((nsamp,nfeat))    
    for i in range(nfeat):
        shuffle(funcs)
        X[:,i] = funcs[0](t + np.random.uniform(0,2))     
    for row in X:
        row+= np.random.normal(0,0.5,nfeat)    
    return X

def syn_mul(nsamp, nfeat, nvariety=3):
    t = np.random.randn(nsamp, nvariety)
    a = np.random.randn(nvariety,nfeat)
    v=np.dot(t,a)
    funcs=[lambda x: x**2, lambda x : log(x), lambda x: x**3, lambda x: x,lambda x: sin(x),lambda x: tan(x)]
    X = np.zeros((nsamp,nfeat))    
    for i in range(nfeat):
        shuffle(funcs)
        X[:,i] = funcs[0](v[:,i])     
    for row in X:
        row+= np.random.normal(0,0.5,nfeat)    
    return X


def error(x,z):
    var=np.var(x,axis=0)
    nfeat=x.shape[1]
    nsamples=x.shape[0]
    return np.mean(np.sum((x - z )**2 , axis=1))  #(var*nfeat)

########################################################################
########################################################################
########################################################################



    
data_source = 'syn_ph'
dataset=syn_ph(1800,200)
missing_percent=np.linspace(0.1,0.9,9)


bjorn_error=[]
mean_error=[]
sd_error=[]
missing_percent=[0.1]
for mis in missing_percent:
    print('missing percentage: ',mis)
    #for i in range(10):
    corruption=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)


    #### SDA
    data_with_missing = dataset * corruption
    gather=Gather_sda(data_with_missing, available_mask = corruption , method = 'nes_mom',dA_initiall = False ,error_known = True )
    
    gather.finetuning()
    #print(train.shape)
    
        
    ###saving result        

    
    bjorn_error.append(sum((1-corruption)*((dataset-gather.gather_out())**2), axis=1).mean())
    mean_error.append(sum((1-corruption)*((dataset-dataset.mean(axis=0))**2), axis=1).mean())
    
#print('gather_out',gather.gather_out()[-1],'outout',sda_result[-1])
#name=data_source+str(datetime.date.today())
#np.savetxt(name,(missing_percent,all_error,known_error,unknown_error))
#missing_percent,all_error,known_error,unknown_error =np.loadtxt('name')

print(bjorn_error)
print(mean_error)

    


"""
#### knn
corrup_knn = corruption.astype('float')
corrup_knn[corrup_knn==0] = np.NAN
knn_data=dataset * corrup_knn
knn_result= knn(knn_data.T,k=10)
"""








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
