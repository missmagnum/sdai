import numpy as np
from pylab import *
import datetime ,gzip, pickle
from gather_sda import Gather_sda

from knn import knn

import time
#%pylab


####data
f = gzip.open('mnist.pkl.gz', 'rb')
(train_set,train_label), (valid_set, vali_label), (test_set, test_labet )= pickle.load(f, encoding='latin1')
f.close()
train_set, valid_set, test_set = 1-train_set, 1-valid_set, 1-test_set  ####Black to white
dataset = train_set


mis=0.6
#for mis in missing_percent:
print('missing percentage: ',mis)


####mask
train_mask = np.random.binomial(n=1, p = 1-mis, size = train_set.shape)
valid_mask = np.random.binomial(n=1, p = 1-mis, size = valid_set.shape)
test_mask = np.random.binomial(n=1, p = 1-mis, size = test_set.shape)
data = (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
mask= train_mask, valid_mask, test_mask

"""
###knn
knn_data = np.split(train_set, 10)
knn_mask = np.split(train_mask, 10)
t0=time.time()
knn_result = knn(knn_data[0] , knn_mask[0],k=50)
tknn=time.time()-t0
"""
###sda
t0=time.time()    
gather=Gather_sda(dataset,data ,problem = 'class', available_mask = mask,
                      method = 'adadelta',
                      pretraining_epochs = 100,
                      pretrain_lr = 0.0005,
                      training_epochs = 100,
                      finetune_lr = 0.0005,
                      batch_size = 400,
                      hidden_size = [1000,1000,100],
                      dA_initiall = True ,
                      error_known = True )
    
gather.finetuning()
tsda=time.time()-t0
#print('time_knn',tknn,'time_sda',tsda)


###plot
subplot(141)
imshow(train_set[200:210].reshape((280, 28)), cmap = cm.Greys_r)
subplot(142)
corrup=train_set[200:210]*train_mask[200:210]
imshow(corrup.reshape((280, 28)), cmap = cm.Greys_r)
subplot(143)
imshow(gather.gather_out()[200:210].reshape((280, 28)), cmap = cm.Greys_r)
subplot(144)
#imshow(knn_result[200:210].reshape((280, 28)), cmap = cm.Greys_r)

show()


   
    
   
