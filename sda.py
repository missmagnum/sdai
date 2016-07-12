import numpy as np
import numpy
import theano
import lasagne

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from perceptron import perceptron



class Sda(object):


    def __init__( self, numpy_rng=None, theano_rng=None,n_inputs=None,
                  hidden_layers_sizes=[500, 500],
                  corruption_levels=[0.1, 0.1]
    ):         

        self.n_layers = len(hidden_layers_sizes)
        self.n_inputs=n_inputs
        self.hidden_layers_sizes=hidden_layers_sizes
        
        assert self.n_layers > 2

        if not numpy_rng:
            numpy_rng = numpy.random.RandomState(123)
               
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')       
        self.x_perd=T.matrix('x_pred')


        ### encoder_layers ####
        
        self.encoder_layers = []
        self.encoder_params = []
        self.dA_layers=[]
        for i in range(self.n_layers):
            
            if i == 0:
                input_size = self.n_inputs
                corruption=True
            else:
                input_size = self. hidden_layers_sizes[i-1]
                corruption=False
          
            if i == 0:
                layer_input = self.x
            else:
                layer_input=self.encoder_layers[-1].output
                
            if i== self.n_layers - 1 :
                act_func=None
            else:
                act_func=T.nnet.sigmoid
                
            self.encoder_layer=perceptron(rng = numpy_rng,
                                          theano_rng=theano_rng,
                                          input = layer_input,
                                          n_in = input_size,
                                          n_out = self.hidden_layers_sizes[i],
                                          activation = act_func,
                                          first_layer_corrup=corruption)

            
            self.encoder_layers.append(self.encoder_layer)
            self.encoder_params.extend(self.encoder_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=self.encoder_layer.W,
                          bhid=self.encoder_layer.b)
            self.dA_layers.append(dA_layer)



        ### decoder_layers ####

        self.decoder_layers = []
        self.decoder_params = []
        
        self.reverse_layers=self.encoder_layers[::-1]
        #self.reverse_da=self.dA_layers[::-1]
        
        decode_hidden_sizes=list(reversed(self.hidden_layers_sizes))

        for i,j in enumerate(decode_hidden_sizes):
            
            
            input_size=j
            if i == 0:
                layer_input=self.reverse_layers[i].output
            else:
                layer_input=self.decoder_layers[-1].output
            
            if i==len(decode_hidden_sizes)-1:
                n_out= self.n_inputs
            else:
                n_out=decode_hidden_sizes[i+1]

            if i== 0:
                act_func=None
            elif i==len(decode_hidden_sizes)-1:
                act_func = T.tanh
            else:
                act_func=T.nnet.relu
            
            self.decoder_layer=perceptron(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=n_out,
                                        W= self.reverse_layers[i].W,
                                        b= None,
                                        activation=act_func,
                                        decoder=True
            )

            
            self.decoder_layers.append(self.decoder_layer)
            #self.decoder_params.append(self.decoder_layer.W)
            self.decoder_params.append(self.decoder_layer.b)

            
        self.network_layers=self.decoder_layers + self.encoder_layers
        self.params = self.encoder_params + self.decoder_params

        
    def out(self):
        numpy_rng = numpy.random.RandomState(123)
        #output=T.dot(self.x, self.decoder_params[-2].get_value().T) + self.decoder_params[-1].get_value()


        output_layer=perceptron(rng=numpy_rng,
                                input = self.decoder_layers[-2].output,
                                n_in = self.hidden_layers_sizes[0],
                                n_out = self.n_inputs,
                                W=self.decoder_layer.W,
                                b=self.decoder_layer.b,                               
                                decoder=True
                                
        )
        return self.decoder_layer.output 
  
       

    def finetune_cost(self):
        numpy_rng = numpy.random.RandomState(123)
        #output=T.dot(self.x, self.decoder_params[-2].get_value().T) + self.decoder_params[-1].get_value()


        output_layer=perceptron(rng=numpy_rng,
                                input = self.decoder_layers[-2].output,
                                n_in = self.hidden_layers_sizes[0],
                                n_out = self.n_inputs,
                                W=self.decoder_layer.W,
                                b=self.decoder_layer.b,                               
                                decoder=True
                                
        )
        #### add regularization
        cost = T.mean(T.sum((self.x - output_layer.output )**2 , axis=0))  

        regularization_l2=lasagne.regularization.apply_penalty(self.params, lasagne.regularization.l2)
        lambda2 = 1e-4
        cost_regu=cost + lambda2 * regularization_l2

        return cost

 
    
       

    def update_method(self, method='nes_mom' , cost= None , params= None, learning_rate= 0.0001):
        
        if method == None :
            gparams = T.grad(finetune_cost, self.params)
            print(gparams)
        
            updates = [
                (param, param - gparam * learning_rate)
                for param, gparam in zip(self.params, gparams)
            ]

        elif method == 'nes_mom' :
            updates = lasagne.updates.nesterov_momentum(self.finetune_cost(),
                                                        self.params,
                                                        learning_rate = learning_rate,
                                                        momentum = 0.9)
        
        elif method == 'adadelta' :
            updates=lasagne.updates.adadelta(self.finetune_cost(),
                                             self.params,
                                             learning_rate = learning_rate,
                                             rho = 0.95,
                                             epsilon = 1e-6)
        else:
            raise ValueError(" check the given UPDATES ")

        return updates




    def pretraining_functions(self, train_set_x, batch_size):

       
        index = T.lscalar('index') 
        corruption_level = T.scalar('corruption')  
        learning_rate = T.scalar('lr')  
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.2),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            
            pretrain_fns.append(fn)

        return pretrain_fns


 
        
    
    def build_finetune_functions(self, method, train_set_x, valid_set_x, test_set_x,
                                 batch_size, learning_rate):
        


        
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index') 

        
        finetune_cost=self.finetune_cost()
        print('finetune:  ',finetune_cost)

        
        updates = self.update_method(method = method,
                                     cost = finetune_cost,
                                     params = self.params)

        train_fn = theano.function(
            inputs=[index],
            outputs = finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            outputs = finetune_cost,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            outputs = finetune_cost,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        out=self.out()
        self.outout=theano.function([],            
            outputs = out,
            givens={
                self.x: train_set_x
            },
            name='train'
        )
            
            


        
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score




