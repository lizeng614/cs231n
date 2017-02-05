# -*- coding: utf-8 -*-
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MyAwesomemodel(object):
    '''
    a experimental convNet with the following architecture:
    
    {conv - [BN] - relu - pool -[Dropout] }xN - conv - [BN] - relu - ...
    [Dropout] - {affine - [BN] - relu - [Dropout]}xM - {affine - relu} - softmax
    
    the first and second {...} block reapeated respectively N and M times
    layers in [] is optional.
    '''
    def __init__(self, FC_dims,input_dim=(3, 32, 32), num_convs=3, num_filters=32, 
                 filter_size=7, num_classes=10, dropout=0, use_batchnorm=False,
                 reg=0.0,weight_scale=1e-3, dtype=np.float64, seed=None):
        """
        Initialize a new FullyConnectedNet.
    
        Inputs:
        - input_dim: An integer giving the size of the input.
        - num_Convs: Number of intermediate convolutional layer. 
        when num_Convs = 0 then there is only the Input layer is convlutional.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer
        - FC_dims: A list of integers giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
        the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
        initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
        this datatype. float32 is faster but less accurate, so you should use
        float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
        will make the dropout layers deteriminstic so we can gradient check the
        model.
        """
        # initialize the parametes
        self.num_convs= num_convs
        self.num_filters= num_filters
        self.filter_size= filter_size
        self.num_FC = len(FC_dims)
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        self.num_layers = num_convs+1+self.num_FC+1
        # initialize the Wi and bi for each layers and save them in self.params{}
            
        # initialze the input Conv layers
        C, H, W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,
                  filter_size)
        self.params['b1'] = weight_scale*np.random.randn(num_filters)
        # initialze the intermediate conv layers
        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)  
        for i in range(1,num_convs+1):
            self.params['W%d' %(i+1)] = weight_scale*np.random.randn(num_filters,
                        num_filters,filter_size,filter_size)
            self.params['b%d' %(i+1)] = np.zeros(num_filters)
            
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %(i+1),'beta%d' %(i+1)
                self.params[gammai] = np.ones(num_filters)
                self.params[betai] = np.zeros(num_filters) 
           
        # initialze the intermedate FC layers
        if type(FC_dims) != list:
            raise ValueError('FC_dims must be a list !')
        num_pool = np.maximum(1,num_convs)
        dims = [num_filters*H*W/4**num_pool] + FC_dims
        for i in range(num_convs+1,num_convs+1+self.num_FC):
            self.params['W%d' %(i+1)] = weight_scale*np.random.randn(dims[i-num_convs-1],
                                                                    dims[i-num_convs])
            self.params['b%d' %(i+1)] = np.zeros(dims[i-num_convs])
           
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %(i+1),'beta%d' %(i+1)
                self.params[gammai] = np.ones(dims[i-num_convs])
                self.params[betai] = np.zeros(dims[i-num_convs]) 
        
        #initialze the last FC layers
        WLast,bLast = 'W%d' %self.num_layers,'b%d' %self.num_layers
        self.params[WLast] = weight_scale*np.random.randn(FC_dims[-1],num_classes)
        self.params[bLast] = np.zeros(num_classes)
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
    
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


            
    def loss(self,X,y = None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
         scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
        """      
        # translate data type  
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode   
        
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.filter_size
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        scores = X.copy()
        cache={}
        cache_DO={}
        
        # forward input conv layers
        Wi,bi = 'W1','b1'
        if self.use_batchnorm:
            gammai,betai = 'gamma1','beta1'
            scores,cache[1] = conv_BN_relu_pool_forward(scores,self.params[Wi],
                        self.params[bi],conv_param,self.params[gammai],
                          self.params[betai],self.bn_params[0],pool_param)
        else:
            scores,cache[1] = conv_relu_pool_forward(scores,self.params[Wi],
                              self.params[bi],conv_param,pool_param)
        if self.use_dropout:
                scores,cache_DO[1] = dropout_forward(scores,self.dropout_param)
        
        # forward through intermediate conv layers
        for i in range(2,self.num_convs+1):
            Wi,bi = 'W%d' %i,'b%d' %i
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %i,'beta%d' %i
                scores,cache[i] = conv_BN_relu_pool_forward(scores,self.params[Wi],
                              self.params[bi],conv_param,self.params[gammai],
                              self.params[betai],self.bn_params[i-1],pool_param)
            else:
                scores,cache[i] = conv_relu_pool_forward(scores,self.params[Wi],
                              self.params[bi],conv_param,pool_param)
            if self.use_dropout:
                scores,cache_DO[i] = dropout_forward(scores,self.dropout_param)
        
        # forward last conv layers 
        nLconv = self.num_convs+1
        if nLconv !=1:
            Wi,bi = 'W%d' %nLconv,'b%d' %nLconv
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %nLconv,'beta%d' %nLconv
                scores,cache[nLconv] = conv_BN_relu_forward(scores,self.params[Wi],
                                self.params[bi],conv_param,self.params[gammai],
                                           self.params[betai],self.bn_params[nLconv-1])
            else:
                scores,cache[nLconv] = conv_relu_forward(scores,self.params[Wi],
                                                   self.params[bi],conv_param)
            if self.use_dropout:
                scores,cache_DO[nLconv] = dropout_forward(scores,self.dropout_param)
        
        # forward intermediate FC layer
        for i in range(nLconv,nLconv+self.num_FC):
            Wi,bi = 'W%d' %(i+1),'b%d' %(i+1)
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %(i+1),'beta%d' %(i+1)
                scores,cache[i+1] = affine_BN_relu_forward(scores,self.params[Wi],
                                        self.params[bi],self.params[gammai],
                                       self.params[betai],self.bn_params[i])
            else:
                scores,cache[i+1] = affine_relu_forward(scores,self.params[Wi],
                                                       self.params[bi])
            if self.use_dropout:
                scores,cache_DO[i+1] = dropout_forward(scores,self.dropout_param)
        
        # forward 4 last FC layer
        nL = self.num_layers
        WLast,bLast = 'W%d' %nL,'b%d' %nL
        # print WLast,bLast
        # print 'nL %d' %nL
        scores,cache_last= affine_relu_forward(scores,self.params[WLast],
                                                       self.params[bLast])
        if mode == 'test':
            return scores
        
        # calc the loss
        data_loss,dout = softmax_loss(scores,y)    
        reg_loss = [[np.sum(self.params['W%d' %(i+1)]**2) for i in range(nL)]]
        reg_loss = 0.5*self.reg*np.sum(reg_loss)
        loss = data_loss + reg_loss   
        
        grads={}
        
        # backprop last FC layer
        dscores, grads[WLast], grads[bLast] = affine_relu_backward(dout,cache_last)
        
        # backprop intermediate FC layers
        for i in reversed(range(nLconv,nLconv+self.num_FC)):
            Wi,bi = 'W%d' %(i+1),'b%d' %(i+1)
            if self.use_dropout:
                gammai,betai = 'gamma%d' %(i+1),'beta%d' %(i+1)
                dscores = dropout_backward(dscores,cache_DO[i+1])
            if self.use_batchnorm:
                dscores,grads[Wi], grads[bi], grads[gammai], grads[betai] = \
                                   affine_BN_relu_backward(dscores,cache[i+1])
            else:
                dscores,grads[Wi], grads[bi] = affine_relu_backward(dscores,cache[i+1])
        
        # backprop last Conv layer
        if nLconv !=1: 
            if self.use_dropout:
               dscores = dropout_backward(dscores,cache_DO[nLconv])
        
            Wi,bi = 'W%d' %nLconv,'b%d' %nLconv
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %nLconv,'beta%d' %nLconv
                dscores,grads[Wi], grads[bi], grads[gammai], grads[betai] = \
                                     conv_BN_relu_backward(dscores,cache[nLconv])
            else:
                dscores,grads[Wi], grads[bi] = conv_relu_backward(dscores,cache[nLconv])
        
        # backprop  the intermedate conv layers
        for i in reversed(range(2,self.num_convs+1)):
            Wi,bi = 'W%d' %i,'b%d' %i
            if self.use_dropout:
                dscores = dropout_backward(dscores,cache_DO[i])
            if self.use_batchnorm:
                gammai,betai = 'gamma%d' %i,'beta%d' %i
                dscores,grads[Wi], grads[bi], grads[gammai], grads[betai] = \
                                  conv_BN_relu_pool_backward(dscores,cache[i])
            else:
                dscores,grads[Wi], grads[bi] = conv_relu_pool_backward(dscores,cache[i])
        
        # backprop input conv layer
        Wi,bi = 'W1','b1'
        if self.use_dropout:
            dscores = dropout_backward(dscores,cache_DO[1])
        if self.use_batchnorm:
            gammai,betai = 'gamma1','beta1'
            dscores,grads[Wi], grads[bi], grads[gammai], grads[betai] = \
                                  conv_BN_relu_pool_backward(dscores,cache[1])
        else:
            dscores,grads[Wi], grads[bi] = conv_relu_pool_backward(dscores,cache[1])
            
        for i in range(nL):
            Wi = 'W%d' %(i+1)
            grads[Wi] += self.reg*self.params[Wi]
              
        return loss,grads
        

