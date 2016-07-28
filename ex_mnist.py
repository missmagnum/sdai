import numpy as np
from pylab import *
import datetime ,gzip, pickle
from gather_sda import Gather_sda

#%matplotlib
#imshow(train_x[:10].reshape((280, 28)), cmap = cm.Greys_r)


         
f = gzip.open('mnist.pkl.gz', 'rb')
(train_set,train_label), (valid_set, vali_label), (test_set, test_labet )= pickle.load(f, encoding='latin1')
f.close()

#imshow(train_set[:10].reshape((280, 28)), cmap = cm.Greys_r)
#show()

dataset = train_set
print(train_set.shape)

missing_percent=[0.1]
for mis in missing_percent:
    print('missing percentage: ',mis)

    train_mask = np.random.binomial(n=1, p = 1-mis, size = train_set.shape)
    valid_mask = np.random.binomial(n=1, p = 1-mis, size = valid_set.shape)
    test_mask = np.random.binomial(n=1, p = 1-mis, size = test_set.shape)
    #### SDA

    data = (train_set*train_mask, valid_set *valid_mask ,test_set *test_mask)
    mask= train_mask, valid_mask, test_mask
     
    gather=Gather_sda(dataset,data ,problem = 'class', available_mask = mask,
                      method = 'nes_mom',
                      pretraining_epochs = 2,
                      pretrain_lr = 0.5,
                      training_epochs = 100,
                      finetune_lr = 0.0005,
                      batch_size = 100,
                      hidden_size = [800,200,200],
                      dA_initiall = True ,
                      error_known = True )
    
    gather.finetuning()
    #print(train.shape)


    
   
