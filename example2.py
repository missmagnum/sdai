import numpy as np
from pylab import *
import datetime

from gather_sda import Gather_sda

#%matplotlib
#imshow(train_x[:10].reshape((280, 28)), cmap = cm.Greys_r)





missing_percent=[0.1]
for mis in missing_percent:
    print('missing percentage: ',mis)
    #for i in range(10):
    #corruption=np.random.binomial(n=1, p = 1-mis, size = dataset.shape)


    #### SDA
    
    gather=Gather_sda( method = 'nes_mom',dA_initiall = True ,error_known = False )
    
    gather.finetuning()
    #print(train.shape)


    
            
            f = gzip.open('mnist.pkl.gz', 'rb')
            (train_set,train_label), (valid_set, vali_label), (test_set, test_labet )= pickle.load(f, encoding='latin1')
            f.close()
            self.train_set=load_data(train_set)
            self.test_set=load_data(test_set)
            self.valid_set=load_data(valid_set)
            imshow(train_set[:10].reshape((280, 28)), cmap = cm.Greys_r)
            show()
