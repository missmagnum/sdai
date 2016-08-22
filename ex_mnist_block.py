import numpy as np
import numpy
from pylab import *
import datetime ,gzip, pickle
from gather_sda import Gather_sda

from knn import knn

import time
#%pylab


def mnist_block(train_set, valid_set, test_set, knn_data, mis):

    dataset = train_set
    n=int(mis*28)
 
    ###mask
    train_mask=np.ones_like(train_set)
    valid_mask=np.ones_like(valid_set)
    test_mask=np.ones_like(test_set)

    block=[0]*28
    for row in range(train_mask.shape[0]):
        ran=np.random.randint(100,700,size=n)
        for r in ran:
            train_mask[row,r:r+28]=block



    data = (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask



    ###knn
    knn_mask = np.split(train_mask, 10)[0]
    t0=time.time()
    knn_result = knn(knn_data , knn_mask,k=50)
    tknn=time.time()-t0


    ###sda
    t0=time.time()    
    gather=Gather_sda(dataset,data ,problem = 'class', available_mask = mask,
                          method = 'nes_mom',
                          pretraining_epochs = 10,
                          pretrain_lr = 0.0005,
                          training_epochs = 100,
                          finetune_lr = 0.0005,
                          batch_size = 200,  ###300
                          hidden_size = [1000,1000,100],
                          dA_initiall = True ,
                          error_known = True )

    gather.finetuning()
    tsda=time.time()-t0
    print('time_knn',tknn,'time_sda',tsda)

    sda_er = np.mean(np.sum((1-train_mask)*((train_set-gather.gather_out())**2), axis=1))
    kn_er = np.mean(np.sum((1-knn_mask)*((knn_data-knn_result)**2), axis=1))
    
    return(sda_er, kn_er)



    
    """
    ###plot
    subplot(141)
    imshow(train_set[200:210].reshape((280, 28)), cmap = cm.Greys_r)
    title('sample')
    subplot(142)
    corrup=train_set[200:210]*train_mask[200:210]
    imshow(corrup.reshape((280, 28)), cmap = cm.Greys_r)
    subplot(143)
    imshow(gather.gather_out()[200:210].reshape((280, 28)), cmap = cm.Greys_r)
    subplot(144)
    imshow(knn_result[200:210].reshape((280, 28)), cmap = cm.Greys_r)
    show()
    """

   
  
if __name__ == "__main__":
    ###data
    f = gzip.open('mnist.pkl.gz', 'rb')
    (train_set,train_label), (valid_set, vali_label), (test_set, test_labet )= pickle.load(f,
                                                                                           encoding='latin1')
    f.close()
    train_set, valid_set, test_set = 1-train_set, 1-valid_set, 1-test_set  ####Black to white
    knn_data = np.split(train_set, 10)[0]
 
    sda_error=[]   
    knn_error=[]
    missing_percent=np.linspace(0.,0.9,10)


    missing_percent=[0.9]
    for mis in missing_percent:
        print('missing percentage: ',mis)

        

        sd,kn=mnist_block(train_set, valid_set, test_set, knn_data, mis)
        sda_error.append(sd)
        knn_error.append(kn)
        #plot(mis,sd,'ro')
        #plot(mis,kn,'g+')
        print (sd,kn)

        
    """
    missing_percent=np.linspace(0.1,0.9,9)
    sda_error = [ 0.736476 , 1.85685, 3.00893, 4.12659 , 5.60219 , 6.30348 , 8.02204 , 9.71651 ,11.1634]
    knn_error = [1.96055 , 4.74457 , 7.46039,  10.5575 , 13.9904 , 16.8772 ,  21.7978, 26.9399,31.4934 ]
    
    plot(missing_percent,sda_error,'r',label='sda')
    plot(missing_percent,knn_error,'g',label='knn')
    xlabel('corruption percentage')
    ylabel('MSE')
    title('dataset: mnist')
    legend(loc=4,prop={'size':9})
    show()
    """
        
