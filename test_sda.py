import numpy
from pylab import *
import timeit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from perceptron import perceptron
from sda import Sda




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


    
def load_data(X):
    #corruption_level = 0.1
    #X= numpy.random.binomial(n = 1 ,
    #p = 1 - corruption_level,
    #size=x.shape) * x
    try:
        matrix = X.as_matrix()
    except AttributeError:
        matrix = X

    
    shared_x = theano.shared(numpy.asarray(matrix,
                dtype=theano.config.floatX), borrow=True)

    return shared_x
    
        




def test_SdA(nfeature,finetune_lr=0.001, pretraining_epochs=3,
             pretrain_lr=0.01, training_epochs=100,
             dataset=None, batch_size=1):

    
    if not dataset:
        dataset=syn_ph(nsamp=1000,nfeat=nfeature)

    n_visible = dataset.shape[1]    
    train_set_x= load_data(dataset)

    
    print('input size',n_visible)
    

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    sda= Sda(
        numpy_rng = numpy_rng,
        theano_rng=theano_rng,
        n_inputs = n_visible,
        hidden_layers_sizes=[600, 200, 2],
        corruption_levels=[0.2, 0.1, 0.1]
    )
    
   
    print('... getting the pretraining functions')
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    corruption_levels = [.3, .2, .1]

    for i in range(sda.n_layers):
        
        for epoch in range(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()

  
 
    ########################
    # FINETUNING THE MODEL #
    ########################
    
   
    valid_set_x = load_data(syn_ph(nsamp=100,nfeat=nfeature))
    test_set_x = load_data(syn_ph(nsamp=100,nfeat=nfeature))
        
    
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        train_set_x=train_set_x,
        valid_set_x=valid_set_x,
        test_set_x=test_set_x,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    ### hold out cross validation
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            #print('boz',minibatch_index,n_train_batches)
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    print('W',sda.decoder_layer.W.get_value()[-1,-1])

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
 
    return sda ,dataset

def mse(x,z):
    return list(((x - z) ** 2).mean(axis=0))
    
    

    
if __name__ == '__main__':
    nfeat=200
    sda,input=test_SdA(nfeat)
    output=sda.outout()
    t = np.linspace(0,2*np.pi,nfeat)
    error=mse(input,output)
 

